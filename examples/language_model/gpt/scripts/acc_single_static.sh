set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="gpt-acc-single"
rm -rf output/$task_name/log

PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch \
    --gpus "6" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-small-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 8\
    --sharding_degree 1\
    --mp_degree 1 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding false \
    --use_amp false \
    --use_recompute false \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 10 \
    --save_steps 100000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 1000 \
    --device "gpu"

 # Not support pipeline for this version, don't change pp_degree.
