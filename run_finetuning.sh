#!/bin/bash

python main.py \
--do_train --do_eval \
--source_lang='en' --target_lang='cm' \
--output_dir='./models/mt5_cmi_freeze_style' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='./data/cmi_control_train_vector.tsv' \
--validation_file='./data/cmi_control_dev_vector.tsv' \
--num_train_epochs=30.0 \
--learning_rate=5e-4 \
--eval_steps=1000 \
--save_steps=1000 \
--evaluation_strategy='steps' \
--save_strategy='steps' 