#!/bin/bash
for train_set_size in 10 20 30 40 50 60 70 80 90 100
do
	if [[ $train_set_size -lt 50 ]]
	then
		num_train_epochs=10.0
		steps=200
	else
		num_train_epochs=20.0
		steps=500
	fi

	train_samples=$((train_set_size*21422/100))
	echo "Running for train set size : ${train_samples}"
	python main_trainer.py \
	--do_train --do_eval --do_predict \
	--source_lang='en' --target_lang='cm' \
	--model_name_or_path='models/mt5_cmgen' \
	--output_dir="models/few_shot/decoder_last/cold_start_mt5_cmi_vector/train_${train_set_size}" \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--gradient_accumulation_steps=2 \
	--overwrite_output_dir=False \
	--predict_with_generate \
	--train_file='data/cmi_vector/hi_cm_train.tsv' \
	--validation_file='data/cmi_vector/hi_cm_valid.tsv' \
	--test_file='data/cmi_vector/hi_cm_test.tsv' \
	--load_best_model_at_end \
	--metric_for_best_model='cmi_bleu_hm' \
	--num_train_epochs=${num_train_epochs} \
	--learning_rate=5e-4 \
	--eval_steps=${steps} \
	--save_steps=${steps} \
	--evaluation_strategy='steps' \
	--save_strategy='steps' \
	--lr_scheduler_type='constant' \
	--generation_num_beams=1 \
	--generation_max_length=128 \
	--optim='adafactor' \
	--max_source_length=128 \
	--max_target_length=128 \
	--save_total_limit=1 \
	--max_train_samples=${train_samples}
done
