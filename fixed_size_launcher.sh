#!/bin/bash

# ----- Parameters -----
# command parameters
dataset='cifar10'
n_layers=10
hidden_size=256
synchronous='true'
accumulation_steps=1
lr=0.1

# testing a single job
for dataset in 'cifar10' 'cifar100'; do
  for accumulation_steps in 1 2 4 8 16; do
    for lr in 0.1 0.05; do
      for hidden_size in 64 128 256; do
        for synchronous in 'true' 'false'; do
          command="sbatch fixed_size_script.sh $dataset $n_layers $hidden_size $synchronous $accumulation_steps $lr"
          echo $command
          eval $command
        done
      done
    done
  done
done
