#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List

import datasets

import transformers
from transformers import (
    T5Tokenizer, T5Config,
    HfArgumentParser,
    Seq2SeqTrainer,
    AdamW,
    Adafactor,
    Seq2SeqTrainingArguments,
    set_seed,
    SchedulerType,
    get_scheduler
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_pt_utils import get_parameter_names

import prepare_dataset as pd
import metrics as mt
from model import MT5ForStyleConditionalGeneration
from classifier_model import ClassifierModel
from trainer_eval import CustomSeq2SeqTrainer

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="google/mt5-small",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    with_tracking: bool = field(
        default=False,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines)."
        })
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    train_file_classify: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines)."
        })
    validation_file_classify: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file_classify: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    max_train_classify_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_classify_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_classify_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )
    num_attr: int = field(
        default=1,
        metadata={
            "help": "Number of sttributes to control for"
        },
    )
    attr_names: str = field(
        default="cmi",
        metadata={
            "help": "Space separated attribute names"
        },
    )


    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.attr_names = data_args.attr_names.split()
    model_name = training_args.output_dir.split('/')[-1]
    log_file = model_name+'.log'
    log_fh = logging.FileHandler(log_file)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), log_fh],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}, "
        + f"place model on device: {training_args.place_model_on_device}, LR : {training_args.learning_rate}, "
        + f"warmup steps : {training_args.warmup_steps}, label_smoothing factor : {training_args.label_smoothing_factor}"
    )
    # training_args.eval_batch_size=8
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    #Load model for style conditional generation.
    logging.info(f"Loading available weights from : {model_args.model_name_or_path}")
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = MT5ForStyleConditionalGeneration.from_pretrained(model_args.model_name_or_path, num_attr=data_args.num_attr)
    classification_model = ClassifierModel(model.style_vector, data_args.num_attr, 512) ##TODO: remove hardcoding
    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if not(training_args.do_train or training_args.do_eval or training_args.do_predict):
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    raw_datasets_generate, raw_datasets_classify = pd.load_data(data_args, model_args)
    # print(raw_datasets["train"])

    if training_args.do_train:
        train_dataset_generate = pd.create_dataset(raw_datasets_generate, data_args, training_args, tokenizer)
        train_dataset_classify = pd.create_dataset_classify(raw_datasets_classify, data_args, training_args, tokenizer)

    if training_args.do_eval:      
        eval_dataset_generate = pd.create_dataset(raw_datasets_generate, data_args, training_args, tokenizer, mode='validation')
        eval_dataset_classify = pd.create_dataset_classify(raw_datasets_classify, data_args, training_args, tokenizer, mode='validation')

    if training_args.do_predict:
        predict_dataset_generate = pd.create_dataset(raw_datasets_generate, data_args, training_args, tokenizer, mode='test')
        predict_dataset_classify = pd.create_dataset_classify(raw_datasets_classify, data_args, training_args, tokenizer, mode='test')

    # Data collator
    data_collator_generate = pd.create_collator_generate(data_args, training_args, tokenizer, model)
    data_collator_classify = pd.create_collator_classify(tokenizer, training_args)


    train_dataloader_generate = DataLoader(
        train_dataset_generate, shuffle=True, collate_fn=data_collator_generate, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader_generate = DataLoader(eval_dataset_generate, collate_fn=data_collator_generate, batch_size=training_args.per_device_eval_batch_size)

    train_dataloader_classify = DataLoader(
        train_dataset_classify, shuffle=True, collate_fn=data_collator_classify, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader_classify = DataLoader(eval_dataset_classify, collate_fn=data_collator_classify, batch_size=training_args.per_device_eval_batch_size)

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_generate = Adafactor(optimizer_grouped_parameters, lr=training_args.learning_rate, scale_parameter=False, relative_step=False)

    optimizer_classify = AdamW([p for _, p in classification_model.named_parameters()], lr=training_args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader_generate) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler_generate = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer_generate,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    if model_args.with_tracking:
        experiment_config = vars(training_args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_generate)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    criterion = nn.BCEWithLogitsLoss()
    metric_for_best_model = 'eval_'+training_args.metric_for_best_model
    best_checkpoint_so_far = None
    best_metric_so_far = -1

    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()
        if model_args.with_tracking:
            total_loss_generate = 0
            total_loss_classify = 0
        
        for step, batch in enumerate(train_dataloader_generate):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            if model_args.with_tracking:
                total_loss_generate += loss.detach().float()
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader_generate) - 1:
                optimizer_generate.step()
                lr_scheduler_generate.step()
                optimizer_generate.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # set model to eval mode before getting encoder embeddings
            classification_model.train()
            model.eval()
            try:
                batch_classify = next(dataloader_classify_iterator)
            except:
                dataloader_classify_iterator = iter(train_dataloader_classify)
                batch_classify = next(dataloader_classify_iterator)
            with torch.no_grad():
                encoder_outputs = model.encoder(
                    input_ids=batch_classify['input_ids'],
                    attention_mask=batch_classify['attention_mask'],
                    return_dict=True
                )
                hidden_states = encoder_outputs.last_hidden_state
                hidden_states = torch.mean(hidden_states * batch_classify["attention_mask"].unsqueeze(-1), axis=1).squeeze()
            
            outputs = classification_model(hidden_states, batch_classify['input_style_scores'])
            loss = criterion(outputs.squeeze(), batch_classify["labels"])
            if model_args.with_tracking:
                total_loss_classify += loss.detach().float()
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader_classify) - 1:
                optimizer_classify.step()
                optimizer_classify.zero_grad()


            #checkpointing and evaluation
            if (completed_steps % training_args.eval_steps == 0):
                # save checkpoint
                output_dir = f"checkpoint-{completed_steps}"
                logger.info(f"Saving checkpoint to {output_dir}")
                if training_args.output_dir is not None:
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                model.save_pretrained(output_dir, state_dict=model.state_dict())

                #evaluate model
                trainer = CustomSeq2SeqTrainer(
                            model=model,
                            args=training_args,
                            eval_dataset=eval_dataset_generate,
                            tokenizer=tokenizer,
                            data_collator=data_collator_generate,
                            compute_metrics=lambda x:mt.compute_metrics(x, tokenizer, data_args),
                        )
                logger.info(f"*** Evaluate ***")
                metrics = trainer.evaluate()
                max_eval_samples = len(eval_dataset_generate)
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset_generate))

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                #keep track of best model so far, assume greater is better. Will not work for loss.
                if metrics[metric_for_best_model] > best_metric_so_far:
                    best_metric_so_far = metrics[metric_for_best_model]
                    best_checkpoint_so_far = completed_steps
                logger.info(f"Best checkpoint so far : checkpoint-{completed_steps}")


    logger.info("Training complete")
    best_model_dir = os.path.join(training_args.output_dir, f"checkpoint-{best_checkpoint_so_far}")
    model = MT5ForStyleConditionalGeneration.from_pretrained(best_model_dir)
    trainer = CustomSeq2SeqTrainer(
                        model=model,
                        args=training_args,
                        eval_dataset=eval_dataset_generate,
                        tokenizer=tokenizer,
                        data_collator=data_collator_generate,
                        compute_metrics=lambda x:mt.compute_metrics(x, tokenizer, data_args),
                        )
    logger.info(f"*** Predict ***")
    predict_results = trainer.predict(
        predict_dataset_generate,
        metric_key_prefix="predict",
        max_length=data_args.max_target_length,
        num_beams=data_args.num_beams,
        )
    metrics = predict_results.metrics
    max_predict_samples = len(predict_dataset_generate)

    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset_generate))
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    return

if __name__ == "__main__":
    main()
