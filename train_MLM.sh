# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
# MODEL=${2:-"$REPO/azure_ml/pytorch_model.bin"}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
ROMA="romanised"
OUT_DIR=${5:-"$REPO/Results"}
MLM_DATA_FILE=${6:-"$REPO/ishan_data/trial.txt"}
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
# export RANK=0
# export WORLD_SIZE=2
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=80

EPOCH=4
BATCH_SIZE=4 #set to match the exact GLUECOS repo
MAX_SEQ=256

echo "starting custom "

python3.6 $PWD/Code/pretrain.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE \
    --config_name $MODEL   \
    --tokenizer_name  $MODEL \
    --output_dir $REPO/NewSwitch_spanish_try2 \
    --train_data_file $REPO/Code/probe/freqmlm_enes_train.txt \
    --eval_data_file $REPO/Code/probe/freqmlm_enes_eval.txt \
    --mlm \
    --line_by_line \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 20\
    --num_train_epochs $EPOCH\
    --logging_steps 100 \
    --seed 100 \
    --save_steps 240 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --mlm_probability 0.35 
    # --mlm_probability 0.38 0.5 0.42 
    # --mlm_probability_syn 0.38
    # --train_data_file_syn $REPO/Data/MLM/TCS/realCS_numtokens_aligned/all_tokens_tcs_train_.txt \
    # --eval_data_file_syn $REPO/Data/MLM/TCS/realCS_numtokens_aligned/all_tokens_tcs_eval_.txt \
    # --switch_probability 0.3 \
    # --mix_probability 0.5 
    # --mlm_probability_syn 0.37 
    # --load_best_model_at_end
    # --no_cuda
    # --first \
    #--train_data_file_syn $REPO/Data/MLM/TCS/realCS_numtokens_aligned/around_switch_tcs_train_.txt \
    #    --eval_data_file_syn $REPO/Data/MLM/TCS/realCS_numtokens_aligned/around_switch_tcs_eval_.txt \
    # --train_data_file $REPO/Data/MLM/combined/splits/around-switch-bert-train.txt $REPO/Data/MLM/TCS/around-switch-tcs-train.txt $REPO/Data/MLM/GCM/around-switch-gcm-train.txt \

