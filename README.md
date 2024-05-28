# Async-Grad

Your new tool for asynchronous deep learning

## Description

This repository contains the code to reproduce the results presented in "PETRA: Parallel End-to-end Training with Reversible Architectures". (Link available soon)

<div style="text-align: center;">
   <img src="./img/petra.jpg" alt="Standard Delayed Gradient vs PETRA" width="600"/>
</div>

## Requiremets

```sh
pip install -r requirements.txt
```

## Preparing data

If you want to download the ImageNet32 prior to training, you can run:

```sh
python scripts/imagenet32loader.py
```

The data is assumed to be located in the ```./data``` folder. For ImageNet, you will need to provide the path for the dataset:

```sh
python main.py --dataset imagenet --dir PATH_TO_DATASET
```

## Usage

### Dataset
For CIFAR-10 and ImageNet32, the data is assumed to be located in the ```./data``` folder.
The code will automatically download those datasets if they are not on disk.

```sh
python main.py --dataset cifar10
python main.py --dataset imagenet32
```

For ImageNet, you will need to provide the path for the dataset:

```sh
python main.py --dataset imagenet --dir PATH_TO_DATASET
```

### Model

The supported models are ```resnet18```, ```resnet34```, ```resnet50```, ```revnet18```, ```revnet34```, ```revnet50```.
```sh
python main.py --model MODEL
```

### Training method

To train a model using standard backpropagation, you can use:

```sh
python main.py --synchronous --store-vjp
```

To train a model using PETRA, you can use:

```sh
python main.py --remove-ctx-param --remove-ctx-input
```

Note that PETRA only applies to reversible architecture, i.e. the option ```--remove-ctx-input``` will not be effective.
Note also that the option ```--remove-ctx-param``` alone corresponds to the Diversely Stale Parameters (DSP) approach.

### Optimizer

This code supports SGD, Adam and LARS optimizer.

To use the SGD optimizer:
```sh
python main.py --optimizer sgd --lr LEARNING_RATE --momentum MOMENTUM --dampening DAMPENING [--nesterov]
```

To use the LARS optimizer:
```sh
python main.py --optimizer lars --lr LEARNING_RATE --momentum MOMENTUM --dampening DAMPENING [--nesterov]
```

To use the Adam optimizer:
```sh
python main.py --optimizer adam --lr LEARNING_RATE --beta1 BETA1 --beta2 BETA2 [--amsgrad]
```

You can set the weight decay with the option ```--weight-decay WEIGHT_DECAY```. You can also remove weight decay on biases and batch-norm parameters with the option ```--no-bn-weight-decay```.

To perform gradient accumulation:
```sh
python main.py --accumulate-steps ACCUMULATION_STEPS [--accumulation-averaging]
```
The option ```--accumulation-averaging``` is used for averaging the gradients over the accumulation steps.
If you want to the linear scaling rule from Goyal et al., use the option ```--goyal-lr-scaling```.
This will scale the learning rate according to the equation:
```sh
scaled_lr = lr * accumulation_steps * batch_size / 256
```

### Scheduler

This code supports multiple schedulers along with linear warm-up.
```sh
python main.py --max-epoch MAX_EPOCH --warm-up WARM_UP_EPOCHS --scheduler SCHEDULER
```

To use the STEPLR scheduler:
```sh
python main.py --scheduler steplr --lr-decay-milestones MILESTONE_1, MILESTONE_2,... --lr-decay-fact DECAY_FACTOR
```

To use the Polynomial scheduler:
```sh
python main.py --scheduler polynomial
```

To use the Cosine scheduler:
```sh
python main.py --scheduler cosine
```

### Checkpoint

To save a checkpoint after each epoch:
```sh
python main.py --name-checkpoint CHECKPOINT_PATH
```

To resume from a checkpoint:
```sh
python main.py --resume CHECKPOINT_PATH
```

## Reproducing results

1. To reproduce the CIFAR-10 results for revnet18:
    ```sh
    python main.py --no-git --dataset cifar10 --batch-size 64 -p 78 --workers 4 --model resnet18 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0005 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 300 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 150 225
    python main.py --no-git --dataset cifar10 --batch-size 64 -p 78 --workers 4 --model revnet18 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0005 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 300 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 150 225
    python main.py --no-git --dataset cifar10 --batch-size 64 -p 78 --workers 4 --model revnet18 --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0005 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 300 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 150 225
    ```

2. To reproduce the ImageNet32 results for revnet34:
    ```sh
    python main.py --no-git --dataset imagenet32 --batch-size 64 -p 2001 --workers 4 --model resnet34 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    python main.py --no-git --dataset imagenet32 --batch-size 64 -p 2001 --workers 4 --model revnet34 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    python main.py --no-git --dataset imagenet32 --batch-size 64 -p 2001 --workers 4 --model revnet34 --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 2 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    ```
3. To reproduce the ImageNet results for revnet50:
    ```sh
    python main.py --no-git --dataset imagenet --dir [PATH_TO_DATASET] --batch-size 64 -p 2001 --workers 16 --model resnet50 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 4 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    python main.py --no-git --dataset imagenet --dir [PATH_TO_DATASET] --batch-size 64 -p 2001 --workers 16 --model revnet50 --synchronous --store-vjp --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 4 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    python main.py --no-git --dataset imagenet --dir [PATH_TO_DATASET] --batch-size 64 -p 2001 --workers 16 --model revnet50 --remove-ctx-input --remove-ctx-param --optimizer sgd --lr 0.1 --weight-decay 0.0001 --no-bn-weight-decay --nesterov --accumulation-steps 4 --accumulation-averaging --goyal-lr-scaling --scheduler steplr --max-epoch 90 --warm-up 5 --lr-decay-fact 0.1 --lr-decay-milestones 30 60 80
    ```
