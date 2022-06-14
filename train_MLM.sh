# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_TYPE=${2:-bert}
OUT_DIR=${3:-"$REPO/models"}
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

EPOCH=2
BATCH_SIZE=4 #set to match the exact GLUECOS repo
MAX_SEQ=128

echo "starting custom "

python3 $PWD/Bert_MLM.py \
    --model_name_or_path $MODEL \
    --model_type $MODEL_TYPE \
    --config_name $MODEL   \
    --tokenizer_name  $MODEL \
    --output_dir $OUT_DIR/MLM_pretrain_opensub_thresh \
    --train_data_file $REPO/data/OpenSubtitles/OpenSubtitles.cmgen.cs \
    --mlm \
    --line_by_line \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCH\
    --seed 100 \
    --save_total_limit 1 \
    --overwrite_output_dir 

