#!/bin/bash

#SBATCH --job-name=petra
#SBATCH --partition=hard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

conda_env="petra-12.4"
echo $conda_env
source activate $conda_env
conda info --envs

# ----- Parameters -----

# command parameters
dataset=$1
n_layers=$2
hidden_size=$3
synchronous=$4
accumulation_steps=$5
lr=$6
# -------------------------------------
wandb_project="iclr2025-async-fully-reversible"
output_dir="logs/$wandb_project" # output directory for logs and checkpoints
# -------------------------------------

# ----- Display parameters -----
echo "dataset: $dataset"
echo "n_layers: $n_layers"
echo "hidden_size: $hidden_size"
echo "synchronous: $synchronous"
echo "accumulation_steps: $accumulation_steps"
echo "lr: $lr"
echo "wandb_project: $wandb_project"
echo ""

# ----- Filename -----
filename="${dataset}-n_layers_${n_layers}-hidden_size_${hidden_size}-sync_${synchronous}-acc_steps_${accumulation_steps}-lr_${lr}"

# ----- Creating logfile and checkpoint -----
mkdir -p "$output_dir"
logfile="${output_dir}/$filename.log"

checkpoint_dir="${output_dir}/checkpoints"
mkdir -p "$checkpoint_dir"
checkpoint="${checkpoint_dir}/$filename.pth"

# ----- Building command -----
command="python -u main_fixed_size.py"
command="${command} --use-wandb --wandb-project $wandb_project"
command="${command} --name-checkpoint $checkpoint --resume $checkpoint"

# dataset
batch_size=64
if [ "$dataset" == 'cifar10' ] || [ "$dataset" == 'cifar100' ]; then
  printf=$((50000 / batch_size / 10))
  workers=4
  command="${command} --dataset $dataset --batch-size $batch_size -p $printf --workers $workers"
elif [ "$dataset" == 'imagenet32' ]; then
  printf=$((1281167 / batch_size / 10))
  workers=8
  command="${command} --dataset imagenet32 --batch-size $batch_size -p $printf --workers $workers --dir ./data/imagenet32"
elif [ "$dataset" == 'imagenet' ]; then
  printf=$((1281167 / batch_size / 10))
  workers=16
  command="${command} --dataset imagenet --batch-size $batch_size -p $printf --workers $workers --dir /gpfsdswork/dataset/imagenet"
fi

# model
command="${command} --n-layers $n_layers"
command="${command} --hidden-size $hidden_size"

# training
if [ "$synchronous" == 'true' ]; then
  command="${command} --synchronous"
fi

# optimization
lr=0.1
command="${command} --optimizer sgd --lr $lr"
if [ "$dataset" == 'cifar10' ] || [ "$dataset" == 'cifar100' ]; then
  command="${command} --weight-decay 0.0005"
elif [ "$dataset" == 'imagenet32' ] || [ "$dataset" == 'imagenet' ]; then
  command="${command} --weight-decay 0.0001"
fi

command="${command} --no-bn-weight-decay"
command="${command} --nesterov"

# gradient computation
command="${command} --remove-ctx-input"
command="${command} --remove-ctx-param"

command="${command} --accumulation-steps $accumulation_steps"
command="${command} --accumulation-averaging"
command="${command} --goyal-lr-scaling"

# scheduling
if [ "$dataset" == 'cifar10' ] || [ "$dataset" == 'cifar100' ]; then
  command="${command} --scheduler steplr --max-epoch 250 --warm-up 15 --lr-decay-fact 0.1 --lr-decay-milestones 150 225"
elif [ "$dataset" == 'imagenet32' ] || [ "$dataset" == 'imagenet' ]; then
  command="${command} --scheduler steplr --max-epoch 90 --warm-up 15 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80"
fi

# ----- Running command -----
echo "logfile: $logfile"
echo ""
echo "checkpoint: $checkpoint"
echo ""
echo "command: $command" | tee -a "$logfile"
echo "" | tee -a "$logfile"
eval "$command" | tee -a "$logfile"
