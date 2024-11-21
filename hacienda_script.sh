#!/bin/bash

#SBATCH --job-name=petra
#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# ----- Parameters -----
# job parameters
gpu_type=$1     # 'a100', 'v100', 'v100-16g', 'v100-32g'
output_dir=$2   # output directory for logs and checkpoints

# command parameters
dataset=$3
model=$4
synchronous=$5
store_vjp=$6
store_input=$7
store_param=$8
approximate_input=$9
accumulation_steps=${10}
lr=${11}
batch_size=${12}
wandb_project=${13}
# ---------------------
filename="${dataset}-${model}-sync_${synchronous}-vjp_${store_vjp}-input_${store_input}-param_${store_param}-approx_input_${approximate_input}-acc_steps_${accumulation_steps}-lr_${lr}-bs_${batch_size}"

# ----- Display parameters -----
echo "gpu_type: $gpu_type"
echo "output_dir: $output_dir"
echo "dataset: $dataset"
echo "model: $model"
echo "store_vjp: $store_vjp"
echo "synchronous: $synchronous"
echo "store_input: $store_input"
echo "store_param: $store_param"
echo "approximate_input: $approximate_input"
echo "accumulation_steps: $accumulation_steps"
echo "lr: $lr"
echo "wandb_project: $wandb_project"
echo "filename: $filename"
echo ""

# ----- Creating logfile and checkpoint -----
output_dir=$2
mkdir -p "$output_dir"
logfile="${output_dir}/$filename.log"

checkpoint_dir="${output_dir}/checkpoints"
mkdir -p "$checkpoint_dir"
checkpoint="${checkpoint_dir}/$filename.pth"

# ----- Building command -----
command="python -u main_error_tracking.py"
command="${command} --use-wandb --wandb-project $wandb_project"
#command="${command} --name-checkpoint $checkpoint --resume $checkpoint"

# dataset
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
command="${command} --model $model"

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
if [ "$store_vjp" == 'true' ]; then
  command="${command} --store-vjp"
fi
if [ "$store_input" == 'false' ]; then
  command="${command} --remove-ctx-input"
fi
if [ "$store_param" == 'false' ]; then
  command="${command} --remove-ctx-param"
fi
if [ "$approximate_input" == 'true' ]; then
  command="${command} --approximate-input"
fi
command="${command} --accumulation-steps $accumulation_steps"
command="${command} --accumulation-averaging"
command="${command} --goyal-lr-scaling"


# scheduling
if [ "$dataset" == 'cifar10' ] || [ "$dataset" == 'cifar100' ]; then
  command="${command} --scheduler steplr --max-epoch 300 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 150 225"
elif [ "$dataset" == 'imagenet32' ] || [ "$dataset" == 'imagenet' ]; then
  command="${command} --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80"
fi


# ----- Running command -----
echo "logfile: $logfile"
echo ""
echo "checkpoint: $checkpoint"
echo ""
echo "command: $command" | tee -a "$logfile"
echo "" | tee -a "$logfile"
eval "$command" | tee -a "$logfile"
