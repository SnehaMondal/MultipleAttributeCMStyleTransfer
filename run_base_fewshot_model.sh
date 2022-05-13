#!/bin/bash

python train_mt5_base_fewshot.py \
--do_train --do_eval --do_test \
--source_lang='en' --target_lang='cm' \
--output_dir='./models/few_shot/base' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--predict_with_generate \
--train_file='./data/few_shot_training/base_train.tsv' \
--validation_file='./data/few_shot_training/base_dev.tsv' \
--test_file=./data/few_shot_training/base_test.tsv \
--load_best_model_at_end \
--metric_for_best_model='loss' \
--num_train_epochs=1.0 \
--learning_rate=5e-4 \
--eval_steps=10 \
--save_steps=10 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--generation_max_length=128 \
--optim='adafactor' \
--max_source_length=128 \
--max_target_length=128 \
--max_train_samples=100