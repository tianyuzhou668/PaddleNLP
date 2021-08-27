set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="ernie-1.0-large-seq128"
rm -rf output/$task_name/log


PYTHONPATH=../../../  python -u  -m paddle.distributed.launch \
    --gpus "4,5" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-large" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 128 \
    --micro_batch_size 32 \
    --global_batch_size 64 \
    --sharding_degree 1\
    --dp_degree 2 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 3960000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu"\

# --check_accuracy true\

# NOTE: please set use_sharding=True for sharding_degree > 1
