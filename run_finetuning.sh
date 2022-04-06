#!/bin/bash

python main.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--output_dir='models/finetune_hd_3' \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--predict_with_generate \
--train_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_train_vector.tsv' \
--validation_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_dev_vector.tsv' \
--test_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_test_vector.tsv' \
--load_best_model_at_end \
--metric_for_best_model='loss' \
--num_train_epochs=5.0 \
--learning_rate=5e-4 \
--eval_steps=1000 \
--save_steps=1000 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--optim='adafactor'
