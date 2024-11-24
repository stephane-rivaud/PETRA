#!/bin/bash

# ----- Parameters -----
# job parameters
gpu_type='a100'  # 'a100', 'v100', 'v100-16g', 'v100-32g'
output_dir='logs/iclr2025-async-test'  # output directory for logs and checkpoints

# command parameters
dataset='cifar10'
n_layers=5
hidden_size=512
synchronous='true'
store_vjp='false'
store_input='false'
store_param='false'
approximate_input='false'
accumulation_steps=1
lr=0.1
batch_size=64
wandb_project="iclr2025-async-rebuttal-depth_${n_layers}"

# testing a single job
for dataset in 'cifar10'; do
  for accumulation_steps in 4 16 32; do
    for n_layers in 20 30 40; do
      synchronous='true'
      sbatch hacienda_fixed_size_script.sh $gpu_type $output_dir $dataset $n_layers $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $lr $batch_size $wandb_project

      synchronous='false'
      sbatch hacienda_fixed_size_script.sh $gpu_type $output_dir $dataset $n_layers $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $lr $batch_size $wandb_project
    done
  done
done
