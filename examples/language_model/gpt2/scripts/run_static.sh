unset CUDA_VISIBLE_DEVICES
#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\

task_name="gpt-fix-4_10"
rm -rf output/$task_name/log

PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch --gpus 1 \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type gpt2\
    --model_name_or_path gpt2-small-en \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --max_steps 70000\
    --grad_clip 1.0\
    --save_steps 10000\
    --eval_steps 500\
    --use_recompute true\
    --use_amp true\
    --warmup_rate 0.01\
    --batch_size 32\
    --device gpu
