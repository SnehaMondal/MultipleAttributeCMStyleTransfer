#!/bin/bash

python main.py \
--do_predict \
--model_name_or_path='./models/mt5_cmi_freeze_style/checkpoint-35000' \
--source_lang='en' --target_lang='cm' \
--output_dir='./results/mt5_cmi_freeze_style' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--predict_with_generate \
--test_file='./data/cmi_control_test_vector_oracle.tsv' \
--validation_file='./data/cmi_control_dev_vector.tsv'