#!/bin/bash

mode='grid'
echo "${mode}"
for seed in 0 1 2 3 4
do
	python generate_cs_from_monolingual_cmi_spi.py \
	--input_filepath="./data/lm_training/emnlp_${mode}_cs/emnlp_${mode}_cs_seed_${seed}.tsv" \
	--output_filepath="./data/lm_training/emnlp_${mode}_cs/output_seed_${seed}.txt"
done
