#!/bin/bash

mkdir -p slurm

# ----- Parameters -----
# job parameters
gpu_type='none'  # 'a100', 'v100', 'v100-16g', 'v100-32g'
output_dir='logs/iclr2025-async-rebuttal-true-delay'  # output directory for logs and checkpoints

# command parameters
dataset='cifar10'
model='revnet18'
synchronous='false'
store_vjp='false'
store_input='false'
store_param='false'
approximate_input='false'
accumulation_steps=1
lr=0.1
batch_size=64
wandb_project='iclr2025-async-rebuttal'

# testing a single job
for dataset in 'cifar10' 'cifar100'; do
  for accumulation_steps in 1 4 16; do
    sbatch hacienda_script.sh $gpu_type $output_dir $dataset $model $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $lr $batch_size $wandb_project
  done
done

# sbatch hacienda_script.sh $gpu_type $output_dir $dataset $model $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $lr $batch_size $wandb_project
