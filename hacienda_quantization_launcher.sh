#!/bin/bash

#!/bin/bash

# Define the function
sbatch_arguments() {
  local dataset=$1
  local model=$2

  # Perform some operations (example: concatenate arguments)
  local partition="jazzy"
  local time="00:20:00"

  # Output the results
  echo "$partition"
  echo "$time"
}

# ----- Create output directory -----

mkdir -p slurm

# ----- Parameters -----

# job parameters
gpu_type='none'                           # 'a100', 'v100', 'v100-16g', 'v100-32g'
output_dir='logs/iclr2025-async-rebuttal' # output directory for logs and checkpoints

# command parameters
dataset='cifar10'
model='revnet18'
synchronous='false'
store_vjp='false'
store_input='false'
store_param='false'
approximate_input='false'
accumulation_steps=16
quantize_buffer='false'
lr=0.1
batch_size=64
wandb_project='iclr2025-async-rebuttal-quantization'

# Call the function with multiple arguments
output=$(sbatch_arguments "$dataset" "$model")
echo "$output"

# Read the output into variables
read partition time <<< "$output"

echo "Partition: $partition"
echo "Time: $time"
sbatch \
  --partition=$partition \
  --time=$time \
  hacienda_quantization_script.sh $gpu_type $output_dir $dataset $model $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $quantize_buffer $lr $batch_size $wandb_project

# testing a single job
#for model in 'revnet18' 'revnet34' 'revnet50' 'revnet101'; do
#  for dataset in 'cifar10' 'cifar100'; do
#    for synchronous in 'false' 'true'; do
#      for quantize_buffer in 'true' 'false'; do
#        sbatch hacienda_quantization_script.sh $gpu_type $output_dir $dataset $model $synchronous $store_vjp $store_input $store_param $approximate_input $accumulation_steps $quantize_buffer $lr $batch_size $wandb_project
#      done
#    done
#  done
#done
