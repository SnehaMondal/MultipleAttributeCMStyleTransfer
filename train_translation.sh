#!/bin/bash

python train_translation.py \
--do_predict \
--source_lang='en' --target_lang='hi' \
--model_name_or_path='google/mt5-small' \
--output_dir='models/mt5_trans_cmgen_multitask' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='data/multitask/shuffled_train_annotated.tsv' \
--validation_file='data/multitask/hi_cm_valid_masked_annotated.tsv' \
--test_file='data/multitask/hi_cm_test_annotated.tsv' \
--load_best_model_at_end \
--metric_for_best_model='bleu' \
--num_train_epochs=5.0 \
--learning_rate=5e-5 \
--eval_steps=1000 \
--save_steps=1000 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--generation_max_length=128 \
--optim='adafactor' \
--max_source_length=128 \
--max_target_length=128 \
--save_total_limit=1