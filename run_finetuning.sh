#!/bin/bash

python main.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--output_dir='models/finetune_hd_2' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--predict_with_generate \
--model_name_or_path='models/pretrain_hd/checkpoint-7200' \
--train_file='/home/shreyapathak9515/controllable_codemixing/cm_data/finetune_hd_train_hi_cm_format.tsv' \
--validation_file='/home/shreyapathak9515/controllable_codemixing/cm_data/finetune_hd_dev_hi_cm_format.tsv' \
--test_file='/home/shreyapathak9515/controllable_codemixing/cm_data/finetune_test_hi_cm_format.tsv' \
--load_best_model_at_end \
--metric_for_best_model='loss' \
--num_train_epochs=5.0 \
--learning_rate=5e-4 \
--eval_steps=300 \
--save_steps=300 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=4
