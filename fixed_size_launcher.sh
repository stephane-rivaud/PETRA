#!/bin/bash

# ----- Parameters -----
# job parameters
gpu_type='a100'  # 'a100', 'v100', 'v100-16g', 'v100-32g'
output_dir='logs/iclr2025-async-test'  # output directory for logs and checkpoints

# command parameters
dataset='cifar10'
n_layers=5
synchronous='false'
store_vjp='false'
store_input='false'
store_param='false'
approximate_input='false'
accumulation_steps=1
lr=0.1
batch_size=64
wandb_project='iclr2025-async-test'

# testing a single job
bash fixed_size_script.sh $gpu_type $output_dir $dataset $n_layers $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $lr $batch_size $wandb_project
