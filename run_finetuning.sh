#!/bin/bash

python main.py \
--do_train --do_eval \
--source_lang='en' --target_lang='cm' \
--output_dir='models/translation_pretrained_ft_cmi_vector' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--model_name_or_path='models/formal_parallel_base' \
--train_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_train_vector.tsv' \
--validation_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_dev_vector.tsv' \
--test_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_test_vector_neutral.tsv' \
--metric_for_best_model='loss' \
--num_train_epochs=30.0 \
--learning_rate=5e-4 \
--eval_steps=1000 \
--save_steps=1000 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--generation_max_length=128 \
--optim='adafactor' \
--max_source_length=128 \
--max_target_length=128 
