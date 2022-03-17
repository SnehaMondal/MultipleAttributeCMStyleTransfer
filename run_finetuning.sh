#!/bin/bash

python main.py \
--model_name_or_path='./models/translation/prod_en_hi/checkpoint-600000' \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--output_dir='../models/translation/prod_bt' \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--predict_with_generate \
--train_file='./data/en_hi_parallel/bt_data/train_tagremoved.tsv' \
--validation_file='./data/en_hi_parallel/bt_data/dev_tagremoved.tsv' \
--test_file='./data/en_hi_parallel/bt_data/test_tagremoved.tsv' \
--load_best_model_at_end \
--metric_for_best_model='loss' \
--num_train_epochs=5.0 \
--learning_rate=1e-5 \
--warmup_ratio=0.5 \
--eval_steps=1000 \
--save_steps=1000 \
--label_smoothing_factor=0.1 \
--weight_decay=1e-6 \
--evaluation_strategy='steps' \
--save_strategy='steps'