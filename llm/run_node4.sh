#!/bin/bash

# export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=$(ip a | grep '10.31.10' | awk '{print $7}')
export NCCL_IB_HCA=mlx5_0
export FLAGS_enable_ixdnn_attn=true
python3 -u  -m paddle.distributed.launch --ips=10.31.10.55,10.31.10.69,10.31.10.210,10.31.10.12 --hosts=10.31.10.55 --gpus "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" run_pretrain.py ./llama/pretrain-llama_13b-pp4tp2sd2_stage1.json
