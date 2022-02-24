
export TASK_NAME=$1
export LR=$2
export BS=$3
export EPOCH=$4
export MAX_SEQ_LEN=$5
export CUDA_VISIBLE_DEVICES=$6
export MODEL_PATH=$7

#     --model_type "ernie"  \
#     --model_name_or_path "ernie-1.0-large" \


#     --model_type "roberta"  \
#     --model_name_or_path "roberta-wwm-ext-large" \

PYTHONPATH=../../../../ \
python -u ./run_clue.py \
    --task_name ${TASK_NAME} \
    --model_type "ernie"  \
    --model_name_or_path "ernie-1.0-large" \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 10 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \
