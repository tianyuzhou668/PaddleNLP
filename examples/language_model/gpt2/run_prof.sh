export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
PYTHONPATH=../../../ nsys profile -t cuda,nvtx \
    -o profiles/dygraph_gpt_inf_mask_report \
    --force-overwrite=true --capture-range=cudaProfilerApi --stop-on-range-end=true  \
    python profile_run_pretrain.py --model_type gpt2\
    --model_name_or_path gpt2-small-en\
    --input_dir "./data"\
    --output_dir "output"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 70000\
    --save_steps 10000\
    --logging_steps 20\
    --eval_steps 500\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --batch_size 16\
    --device gpu
