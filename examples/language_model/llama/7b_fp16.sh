# export NCCL_DEBUG=INFO 
# export NCCL_DEBUG_SUBSYS=ALL
export FLAGS_use_cuda_managed_memory=true
rm -rf ./tmp/*
export NCCL_IB_TIMEOUT=22
export FLAGS_cudnn_deterministic=True

PYTHONPATH=../../../:$PYTHONPATH python -u  -m paddle.distributed.fleet.launch \
    --gpus "4"  --log_dir ./tmp finetune_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_micro_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --pipeline_parallel_config "disable_p2p_cache_shape" \
    --eval_with_do_generation 0 \
    --use_flash_attention 1 \
    --overwrite_output_dir \
    --scale_loss 512 \
    --output_dir ./checkpoints/ \
    --logging_steps 1 \
    --disable_tqdm 1 \
    --eval_steps 500 \
    --fp16 1 \
    --fp16_opt_level O2 \
    --recompute 1 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20
