import datasets
from datasets import load_dataset
from transformers import default_data_collator, DataCollatorForSeq2Seq
import torch
import random
random.seed(0)

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
    raw_datasets = load_dataset('csv', data_files=data_files, delimiter = '\t',
                                cache_dir=model_args.cache_dir,
                                column_names=[data_args.source_lang, data_args.target_lang]+["task", "cmi", "spi"])
    return raw_datasets

class CustomDataset(torch.utils.data.Dataset):
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

def preprocess_function(examples, tokenizer, data_args):
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    en_hi_prefix = "to_hi "
    hi_en_prefix = "to_en "
    cm_prefix = "to_cm "

    inputs=[]
    targets=[]
    cmi_scores=[]
    spi_scores=[]

    for inp, tar, task, cmi, spi in zip(examples[data_args.source_lang], examples[data_args.target_lang], examples["task"], examples["cmi"], examples["spi"]):
        if task == "cm":
            inputs.append(cm_prefix + str(inp).strip())
            targets.append(str(tar).strip())
            cmi_scores.append(float(cmi))
            spi_scores.append(float(spi))
        elif task == "trans":
            inputs.append(en_hi_prefix + str(inp).strip())
            targets.append(tar)
            cmi_scores.append(0.0)
            spi_scores.append(0.0)

            inputs.append(hi_en_prefix + str(tar).strip())
            targets.append(inp)
            cmi_scores.append(0.0)
            spi_scores.append(0.0)
        else:
            continue
    assert len(inputs) == len(targets)
    assert len(inputs) == len(spi_scores)

    inputs_and_targets = [(inp, tar, cmi, spi) for\
                        inp, tar, cmi, spi in zip(inputs, targets, cmi_scores, spi_scores) if inp!="" and tar!="" and inp is not None and tar is not None]
    random.shuffle(inputs_and_targets)

    inputs = [inp for (inp, _, _, _) in inputs_and_targets]
    targets = [tar for (_, tar, _, _) in inputs_and_targets]
    cmi_scores = [score for (_, _, score, _) in inputs_and_targets]
    spi_scores = [score for (_, _, _, score) in inputs_and_targets]
    style_scores = [cmi_scores, spi_scores]

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
    return CustomDataset(model_inputs)

def create_dataset(raw_datasets, data_args, training_args, tokenizer, mode='train'):
    if mode=='train':
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = preprocess_function(train_dataset, tokenizer, data_args)
            # train_dataset = train_dataset.map(
            #     preprocess_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     fn_kwargs={'tokenizer':tokenizer, 'data_args':data_args},
            #     desc="Running tokenizer on train dataset",
            # )
        return train_dataset
    elif mode=='validation':
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = preprocess_function(eval_dataset, tokenizer, data_args)
            # eval_dataset = eval_dataset.map(
            #     preprocess_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     fn_kwargs={'tokenizer':tokenizer, 'data_args':data_args},
            #     desc="Running tokenizer on validation dataset",
            # )
        return eval_dataset
    elif mode=='test':
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = preprocess_function(predict_dataset, tokenizer, data_args)
            # predict_dataset = predict_dataset.map(
            #     preprocess_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     fn_kwargs={'tokenizer':tokenizer, 'data_args':data_args},
            #     desc="Running tokenizer on prediction dataset",
            # )
        return predict_dataset
    else: raise ValueError("wrong mode for create_dataset function")

def create_collator(data_args, training_args, tokenizer, model):
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
