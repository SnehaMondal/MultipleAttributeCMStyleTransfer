import datasets
from datasets import load_dataset
from transformers import default_data_collator, DataCollatorForSeq2Seq, DataCollatorWithPadding
import torch
import pickle
from cmi import get_cmi_bucket_tag

prefix = "to_cm "

def load_data(data_args, model_args):
    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets_generate = load_dataset('csv', data_files=data_files, delimiter = '\t',
                                cache_dir=model_args.cache_dir,
                                column_names=[data_args.source_lang, data_args.target_lang]+data_args.attr_names)

    data_files_classify = {}
    raw_datasets_classify = None
    if model_args.use_classification_obj:
        if data_args.train_file_classify is not None:
            data_files_classify["train"] = data_args.train_file_classify
        if data_args.validation_file_classify is not None:
            data_files_classify["validation"] = data_args.validation_file_classify
        if data_args.test_file_classify is not None:
            data_files_classify["test"] = data_args.test_file_classify
        raw_datasets_classify = load_dataset('csv', data_files=data_files_classify, delimiter = '\t',
                                    cache_dir=model_args.cache_dir,
                                    column_names=[data_args.source_lang] + data_args.attr_names + ['labels'])
    return raw_datasets_generate, raw_datasets_classify

class CustomDatasetGenerate(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs['input_ids']
        self.attention = inputs['attention_mask']
        self.targets = inputs['labels']
        self.style = inputs['input_style_scores']
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index]).squeeze()
        attention_mask = torch.tensor(self.attention[index]).squeeze()
        target_ids = torch.tensor(self.targets[index]).squeeze()
        style_scores = self.style[index]
        
        return {"input_ids": input_ids, "labels": target_ids, "attention_mask":attention_mask, "input_style_scores":style_scores}

class CustomDatasetClassify(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs['input_ids']
        self.attention = inputs['attention_mask']
        self.attr = inputs['labels']
        self.style = inputs["input_style_scores"]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index]).squeeze()
        attention_mask = torch.tensor(self.attention[index]).squeeze()
        attr_labels = (self.attr[index]).squeeze()
        style_scores = self.style[index]
        
        return {"input_ids": input_ids, "attention_mask":attention_mask, "labels":attr_labels, "input_style_scores":style_scores}

def preprocess_function_generate(examples, tokenizer, data_args):
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    ### todo: what is the format of examples? fix accordingly
    inputs = [str(source) for source in examples[data_args.source_lang]]
    targets = [str(target) for target in examples[data_args.target_lang]]
    style_scores = [[float(score) for score in examples[attr]] for attr in data_args.attr_names]
    inputs = [prefix + inp for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
#    print(labels)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["input_style_scores"] = torch.tensor(style_scores, dtype=torch.float32)
    model_inputs["input_style_scores"] = torch.transpose(model_inputs["input_style_scores"], 0, 1)
    return CustomDatasetGenerate(model_inputs)

def preprocess_function_classify(examples, tokenizer, data_args):
    # Temporarily set max_target_length for training.
    padding = "max_length" if data_args.pad_to_max_length else False

    ### todo: what is the format of examples? fix accordingly
    inputs = [str(source) for source in examples[data_args.source_lang]]
    labels = [int(x) for x in examples['labels']]
    style_scores = [[float(score) for score in examples[attr]] for attr in data_args.attr_names]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    model_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)  
    model_inputs["input_style_scores"] = torch.tensor(style_scores, dtype=torch.float32)
    model_inputs["input_style_scores"] = torch.transpose(model_inputs["input_style_scores"], 0, 1)
    return CustomDatasetClassify(model_inputs)

def create_dataset(raw_datasets, data_args, training_args, tokenizer, mode='train'):
    if mode=='train':
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        print("Total examples:", len(train_dataset))
        cmi_cutoffs = pickle.load(open(data_args.cmi_cutoffs_dict,'rb'))[3]
        train_dataset = train_dataset.filter(lambda example: get_cmi_bucket_tag(example[data_args.target_lang], cmi_cutoffs)!=data_args.holdout_bucket)
        print("Filtered examples:", len(train_dataset))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = preprocess_function_generate(train_dataset, tokenizer, data_args)
        return train_dataset
    elif mode=='validation':
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = preprocess_function_generate(eval_dataset, tokenizer, data_args)
        return eval_dataset
    elif mode=='test':
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = preprocess_function_generate(predict_dataset, tokenizer, data_args)
        return predict_dataset
    else: raise ValueError("wrong mode for create_dataset function")

def create_dataset_classify(raw_datasets, data_args, training_args, tokenizer, mode='train'):
    if mode=='train':
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_classify_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_classify_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = preprocess_function_classify(train_dataset, tokenizer, data_args)
        return train_dataset
    elif mode=='validation':
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_classify_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_classify_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = preprocess_function_classify(eval_dataset, tokenizer, data_args)
        return eval_dataset
    elif mode=='test':
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_classify_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_classify_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = preprocess_function_classify(predict_dataset, tokenizer, data_args)
        return predict_dataset
    else: raise ValueError("wrong mode for create_dataset function")


def create_collator_generate(data_args, training_args, tokenizer, model):
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    return data_collator


def create_collator_classify(tokenizer, training_args):
        return DataCollatorWithPadding(tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 else None)
