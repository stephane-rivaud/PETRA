#!/bin/bash

# ----- Parameters -----
# command parameters
dataset='cifar10'
n_layers=10
hidden_size=512
synchronous='false'
accumulation_steps=1
lr=0.2

# launching experiments
for dataset in 'cifar10' 'cifar100'; do
  for accumulation_steps in 1 2 4 8 16; do
    for n_layers in 5 10; do
      for hidden_size in 64 128 256; do
        command="sbatch fixed_size_script.sh $dataset $n_layers $hidden_size $synchronous $accumulation_steps $lr"
        echo $command
        eval $command
      done
    done
  done
done
