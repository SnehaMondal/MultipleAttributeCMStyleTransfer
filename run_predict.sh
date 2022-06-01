#!/bin/bash

python main.py \
--do_predict \
--model_name_or_path='./models/mt5_hd_ft_cmi_vector_0106' \
--source_lang='en' --target_lang='cm' \
--output_dir='./results/scaled_vector_cmi' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--predict_with_generate \
--test_file='./cm_data/cmi_control_vec/cmi_control_test_cmi_vector.tsv' \
--validation_file='./cm_data/cmi_control_vec/cmi_control_dev_cmi_vector.tsv'
