#!/bin/bash

python train_mt5_base_fewshot.py \
--do_train --do_eval  \
--source_lang='en' --target_lang='cm' \
--output_dir='./models/formal_parallel_base' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='../cm_data/formal_news_train.csv' \
--validation_file='../cm_data/formal_news_dev.csv' \
--load_best_model_at_end \
--metric_for_best_model='bleu' \
--num_train_epochs=30.0 \
--learning_rate=5e-4 \
--eval_steps=350 \
--save_steps=350 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--generation_max_length=128 \
--optim='adafactor' \
--max_source_length=128 \
--max_target_length=128 \
--save_total_limit=1 \
--source_prefix='to_hi'
