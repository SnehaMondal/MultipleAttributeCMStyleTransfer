#!/bin/bash

python train_translation.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='hi' \
--model_name_or_path='models/translate_pretrained/checkpoint-92000' \
--output_dir='models/translate_pretrained' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='data/translation_data/parallel-n/shuffled_translate_all.parallel' \
--validation_file='data/translation_data/dev_test/dev.parallel' \
--test_file='data/translation_data/dev_test/test.parallel' \
--load_best_model_at_end \
--metric_for_best_model='bleu' \
--num_train_epochs=5.0 \
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
--save_total_limit=1
