#!/bin/bash

python main.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--output_dir='models' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=2 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=True \
--train_file='codemixed_generation_en_cm_train_format.tsv' \
--validation_file='codemixed_generation_en_cm_val_format.tsv' \
--test_file='codemixed_generation_en_cm_test_format.tsv' \
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