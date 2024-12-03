#!/bin/bash

#!/bin/bash

# Define the function
sbatch_arguments() {
  local dataset=$1
  local model=$2

  # Perform some operations (example: concatenate arguments)
  if [ $model == "revnet18" ]; then
    # partition
    local partition="funky"

    # time
    if [ $dataset == "cifar10" ] || [ $dataset == "cifar100" ]; then
      local time="04:30:00"
    elif [ $dataset == "imagenet32" ]; then
      local time="19:00:00"
    fi

  elif [ $model == "revnet34" ]; then
    # partition
    local partition="funky"

    # time
    if [ $dataset == "cifar10" ] || [ $dataset == "cifar100" ]; then
      local time="07:30:00"
    elif [ $dataset == "imagenet32" ]; then
      local time="32:00:00"
    fi

  elif [ $model == "revnet50" ]; then
    local partition="electronic"
    local time="67:00:00"

  elif [ $model == "revnet101" ]; then
    local partition="hard"
    local time="19:00:00"
  fi

  # Output the results
  echo "$partition"
  echo "$time"
}

# ----- Create output directory -----

mkdir -p slurm

# ----- Parameters -----

# job parameters
output_dir='logs/iclr2025-async-quantization' # output directory for logs and checkpoints

# command parameters
dataset='cifar10'
model='revnet18'
synchronous='false'
accumulation_steps=16
quantize_buffer='false'
wandb_project='iclr2025-async-rebuttal-quantization'

# Call the function with multiple arguments
output=$(sbatch_arguments "$dataset" "$model")

# Read the output into variables
IFS=$'\n' read -d '' -r partition time <<< "$output"

# ----- Launch jobs -----
dataset='cifar10'
model='revnet34'
synchronous='false'
quantize_buffer='true'
sbatch \
  --partition=$partition \
  --time=$time \
  hacienda_quantization_script.sh $output_dir $dataset $model $synchronous $accumulation_steps $quantize_buffer $wandb_project

dataset='cifar10'
model='revnet50'
synchronous='true'
quantize_buffer='true'
sbatch \
  --partition=$partition \
  --time=$time \
  hacienda_quantization_script.sh $output_dir $dataset $model $synchronous $accumulation_steps $quantize_buffer $wandb_project


dataset='cifar10'
model='revnet101'
synchronous='true'
for quantize_buffer in 'true' 'false'; do
  sbatch \
    --partition=$partition \
    --time=$time \
    hacienda_quantization_script.sh $output_dir $dataset $model $synchronous $accumulation_steps $quantize_buffer $wandb_project
done

dataset='cifar100'
model='revnet50'
for synchronous in 'true' 'false'; do
  for quantize_buffer in 'true' 'false'; do
    sbatch \
      --partition=$partition \
      --time=$time \
      hacienda_quantization_script.sh $output_dir $dataset $model $synchronous $accumulation_steps $quantize_buffer $wandb_project
  done
done

dataset='cifar100'
model='revnet101'
synchronous='true'
for quantize_buffer in 'true' 'false'; do
  sbatch \
    --partition=$partition \
    --time=$time \
    hacienda_quantization_script.sh $output_dir $dataset $model $synchronous $accumulation_steps $quantize_buffer $wandb_project
done


