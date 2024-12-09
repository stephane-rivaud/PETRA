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
    # partition
    local partition="electronic"

    # time
    if [ $dataset == "cifar10" ] || [ $dataset == "cifar100" ]; then
      local time="13:00:00"
    elif [ $dataset == "imagenet32" ]; then
      local time="67:00:00"
    fi

  elif [ $model == "revnet101" ]; then
    # partition
    local partition="hard"

    # time
    if [ $dataset == "cifar10" ] || [ $dataset == "cifar100" ]; then
      local time="19:00:00"
    elif [ $dataset == "imagenet32" ]; then
      local time="115:00:00"
    fi
  fi

  # Output the results
  echo "$partition"
  echo "$time"
}

# ----- Create output directory -----

mkdir -p slurm

# ----- Parameters -----

# job parameters

# command parameters
dataset='cifar10'         # 'cifar10', 'cifar100', 'imagenet32'
model='revnet18'          # 'revnet18', 'revnet34', 'revnet50', 'revnet101'
synchronous='false'       # 'true', 'false'
accumulation_steps=16     # number of accumulation steps
quantizer='QuantizSimple' # 'QuantizSimple', 'Quantiz8Bits', 'Quantiz16Bits', 'QuantizQSGD'
wandb_project='iclr2025-async-rebuttal-quantization'
output_dir='logs/iclr2025-async-quantization' # output directory for logs and checkpoints


# ----- Launch jobs -----
# RevNet50
dataset='cifar100'
model='revnet50'
synchronous='true'
quantizer='Quantiz8Bits'

output=$(sbatch_arguments "$dataset" "$model")
IFS=$'\n' read -d '' -r partition time <<<"$output"
command="sbatch --partition=$partition --time=$time hacienda_quantization_script.sh $dataset $model $synchronous $accumulation_steps $quantizer $wandb_project $output_dir"
echo "$command"
eval "$command"

synchronous='false'
quantizer='Quantiz16Bits'

output=$(sbatch_arguments "$dataset" "$model")
IFS=$'\n' read -d '' -r partition time <<<"$output"
command="sbatch --partition=$partition --time=$time hacienda_quantization_script.sh $dataset $model $synchronous $accumulation_steps $quantizer $wandb_project $output_dir"
echo "$command"
eval "$command"

# RevNet101
dataset='cifar10'
model='revnet101'
synchronous='false'
for quantizer in 'Quantiz8Bits' 'QuantizQSGD'; do
  output=$(sbatch_arguments "$dataset" "$model")
  IFS=$'\n' read -d '' -r partition time <<<"$output"
  command="sbatch --partition=$partition --time=$time hacienda_quantization_script.sh $dataset $model $synchronous $accumulation_steps $quantizer $wandb_project $output_dir"
  echo "$command"
  eval "$command"
done

dataset='cifar100'
model='revnet101'
synchronous='true'
for quantizer in 'QuantizSimple' 'QuantizQSGD'; do
  output=$(sbatch_arguments "$dataset" "$model")
  IFS=$'\n' read -d '' -r partition time <<<"$output"
  command="sbatch --partition=$partition --time=$time hacienda_quantization_script.sh $dataset $model $synchronous $accumulation_steps $quantizer $wandb_project $output_dir"
  echo "$command"
  eval "$command"
done

synchronous='false'
for quantizer in 'Quantiz8Bits' 'QuantizQSGD'; do
  output=$(sbatch_arguments "$dataset" "$model")
  IFS=$'\n' read -d '' -r partition time <<<"$output"
  command="sbatch --partition=$partition --time=$time hacienda_quantization_script.sh $dataset $model $synchronous $accumulation_steps $quantizer $wandb_project $output_dir"
  echo "$command"
  eval "$command"
done
