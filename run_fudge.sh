#!/bin/bash

# python ./code/run_fudge.py \
# --input_filename='./data/test_fudge.tsv' \
# --output_filename_prefix='./output/fudge/base_predictor/wordpair/translation' \
# --path_to_predictor='xlm-roberta-base' \
# --reference_index=2

python run_fudge.py \
--input_filename='./data/cmi_control_test_vector_oracle.tsv' \
--output_directory='./results/fudge/translation_pretrained_ft_cmi_vector' \
--path_to_cmgen_model='./models/translation_pretrained_ft_cmi_vector/checkpoint-41000' \
--path_to_predictor='xlm-roberta-base' \
--beam_width=4

# python ./code/run_fudge.py \
# --input_filename='./data/en_hi_parallel/ct_data/test_data/tagremoved_creatives_test_data.tsv-00000-of-00001' \
# --output_filename_prefix='./output/fudge/prod_ct_predictor_base/creative_testset/translation' \
# --path_to_predictor='xlm-roberta-base' \
# --reference_index=1

# python ./code/run_fudge.py \
# --input_filename='./data/test_fudge.tsv' \
# --output_filename_prefix='./output/fudge/finetuned_predictor/wordpair/translation' \
# --path_to_predictor='./models/formality_classifiers/predictor/checkpoint-400' \
# --reference_index=2