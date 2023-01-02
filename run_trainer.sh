#!/bin/bash

python main_trainer.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--model_name_or_path='google/mt5-small' \
--output_dir='models/translation_multitask_cmi_spi_vec' \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='data/multitask_cmi_spi_vec/train_annotated.tsv' \
--validation_file='data/multitask_cmi_spi_vec/hi_cm_valid_masked_annotated.tsv' \
--test_file='data/multitask_cmi_spi_vec/hi_cm_test_annotated.tsv' \
--load_best_model_at_end \
--metric_for_best_model='cmi_spi_bleu_hm' \
--num_train_epochs=3.0 \
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
--max_target_length=128 \
--save_total_limit=1 \
--num_attr=2

#### cpi, spi 

# python main.py \
# --do_train --do_eval \
# --source_lang='en' --target_lang='cm' \
# --output_dir='models/mt5_hd_ft_cmi_spi_vector' \
# --per_device_train_batch_size=8 \
# --per_device_eval_batch_size=8 \
# --gradient_accumulation_steps=2 \
# --overwrite_output_dir=False \
# --predict_with_generate \
# --train_file='/home/shreyapathak9515/controllable_codemixing/cm_data/joint_control_train_vector.tsv' \
# --validation_file='/home/shreyapathak9515/controllable_codemixing/cm_data/joint_control_dev_vector.tsv' \
# --metric_for_best_model='loss' \
# --num_train_epochs=10.0 \
# --learning_rate=5e-4 \
# --eval_steps=1000 \
# --save_steps=1000 \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --lr_scheduler_type='constant' \
# --generation_num_beams=1 \
# --generation_max_length=128 \
# --optim='adafactor' \
# --max_source_length=128 \
# --max_target_length=128 \
# --num_attr=2 \
# --attr_names='cmi spi'
