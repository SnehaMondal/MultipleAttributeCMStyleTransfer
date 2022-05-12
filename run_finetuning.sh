#!/bin/bash

python run_finetuning.py \
--do_train --do_eval \
--source_lang='en' --target_lang='cm' \
--source_prefix='to_cm ' \
--output_dir='models' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--predict_with_generate \
--train_file='./data/cmi_control_hd/pretrain_hd_train_hi_cm.tsv' \
--validation_file='./data/cmi_control_hd/pretrain_hd_dev_hi_cm.tsv' \
--load_best_model_at_end \
--metric_for_best_model='loss' \
--num_train_epochs=10.0 \
--learning_rate=5e-4 \
--eval_steps=200 \
--save_steps=200 \
--evaluation_strategy='steps' \
--save_strategy='steps' 