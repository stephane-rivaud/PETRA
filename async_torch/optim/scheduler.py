import math


# without momentum, weight decay and BN: SYNC: train 90.6, test: 83.0

# ASync with lr renorm 1/(2*depth) train 48, test 48


def warm_up_lr(iter, total_iters, lr_final):
    gamma = (iter + 1) / total_iters
    return gamma * lr_final


def adjust_learning_rate_step_lr(epoch, milestones, lr_decay_fact, lr_initial):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    scaling = 0
    for i in range(len(milestones)):
        if epoch >= milestones[-1]:
            scaling = len(milestones)
            break
        elif epoch < milestones[i]:
            scaling = i
            break
    lr = lr_initial * lr_decay_fact ** scaling
    return lr


def adjust_learning_rate_polynomial(epoch, total_epochs, lr_initial):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    gamma = 1.0 - (epoch + 1) / total_epochs
    gamma = gamma ** 2
    lr = lr_initial * gamma
    return lr


def adjust_learning_rate_cosine(epoch, total_epochs, lr_initial, lr_min=0.0):
    lr_t = lr_min + 0.5 * (lr_initial - lr_min) * (1 + math.cos(math.pi * epoch / (total_epochs + 1)))
    return lr_t


def adjust_learning_rate_goyal_original(
        k_step,
        n_step_tot,
        n_epoch_if_1_worker,
        lr,
        world_size,
        batch_size,
        dataset_name
):
    """
    A function returning a function that generate the modified lr schedule for large batch size of Goyal et al. https://arxiv.org/pdf/1706.02677.pdf
    This function is then used in the LambdaLR scheduler of pytorch.

    Parameters:
        - k_step (int): the current value of the iteration counter.
        - n_step_tot (int): the total number of iterations.
        - n_epoch_if_1_worker (int): the total number of epochs.
        - lr (float): the base learning rate.
        - world_size (int): the number of workers.
        - batch_size (int): the batch size per worker.
        - dataset_name (str): the name of the datasets (as scheduler differ for ImageNet and CIFAR10).
        - return_function (bool): whether to return a lambda function, or the value of the multiplicative factor to the base
                                  learning rate at the current step.

    Returns:
        - Either one of the multiplicative factor, or a function to use in LambdaLR, depending on return_function.
    """

    # init of the constants
    n_step_per_epoch = n_step_tot // n_epoch_if_1_worker
    five_epoch_step = 5 * n_step_per_epoch
    milestones = [five_epoch_step]
    multiplicative_factor = 1
    # create the milestones, depending on the dataset
    if dataset_name == "cifar10":
        # put 2*n_step_tot to be sure the last stage lasts all of the remaining of the training
        milestones += [int(0.5 * n_step_tot), int(0.75 * n_step_tot), 2 * n_step_tot]
    else:
        milestones += [
            int(0.3 * n_step_tot),
            int(0.6 * n_step_tot),
            int(0.8 * n_step_tot),
            2 * n_step_tot,
        ]
    # create the linear warm up from base lr to the value of
    # lr x (bs/256) x world_size for the 5 first epochs
    if k_step < five_epoch_step:
        linear_slope = (1 / five_epoch_step) * (batch_size * world_size / 256 - 1)
        multiplicative_factor = 1 + k_step * linear_slope
    else:
        for k in range(len(milestones) - 1):
            if milestones[k] <= k_step < milestones[k + 1]:
                multiplicative_factor = (0.1 ** k) * batch_size * world_size / 256
    return lr * multiplicative_factor


def adjust_learning_rate_goyal(
        k_step,
        n_step_tot,
        n_epoch,
        milestones,
        lr,
        world_size,
        batch_size
):
    """
    A function returning a function that generate the modified lr schedule for large batch size of Goyal et al. https://arxiv.org/pdf/1706.02677.pdf
    This function is then used in the LambdaLR scheduler of pytorch.

    Parameters:
        - k_step (int): the current value of the iteration counter.
        - n_step_tot (int): the total number of iterations.
        - n_epoch_if_1_worker (int): the total number of epochs.
        - lr (float): the base learning rate.
        - world_size (int): the number of workers.
        - batch_size (int): the batch size per worker.
        - dataset_name (str): the name of the datasets (as scheduler differ for ImageNet and CIFAR10).
        - return_function (bool): whether to return a lambda function, or the value of the multiplicative factor to the base
                                  learning rate at the current step.

    Returns:
        - Either one of the multiplicative factor, or a function to use in LambdaLR, depending on return_function.
    """

    # init of the constants
    n_step_per_epoch = n_step_tot // n_epoch
    five_epoch_step = 5 * n_step_per_epoch
    milestones = [five_epoch_step] + [m * n_step_per_epoch for m in milestones] + [n_step_tot + 1]
    multiplicative_factor = 1
    # create the milestones, depending on the dataset

    # create the linear warm up from base lr to the value of
    # lr x (bs/256) x world_size for the 5 first epochs
    if k_step < five_epoch_step:
        linear_slope = (1 / five_epoch_step) * (batch_size * world_size / 256 - 1)
        multiplicative_factor = 1 + k_step * linear_slope
    else:
        for k in range(len(milestones) - 1):
            if milestones[k] <= k_step < milestones[k + 1]:
                multiplicative_factor = (0.1 ** k) * batch_size * world_size / 256
    return lr * multiplicative_factor
