import math

import torch
from torch.nn import init


def init_conv(n_out, n_in, kernel_size):
    """Initialize a convolutional layer according to the kernel parameters.

    Parameters
    ----------
    n_out : int
        the number of channels out
    n_in : int
        the number of channels int
    kernel_size : int
        the width of the kernel

    Returns
    -------
    weight : torch.tensor
        the initialized corresponding convolutional kernel
    bias : torch.tensor
        the corresponding bias
    """
    weight = torch.empty(n_out, n_in, kernel_size, kernel_size)
    bias = torch.empty(n_out)

    init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
    if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)
    return weight, bias


def init_batchnorm(n_out):
    """Initialize a batch normalization layer.

    Parameters
    ----------
    n_out : int
        the number of channels out

    Returns
    -------
    running_var : torch.tensor
        the corresponding running var
    running_mean : torch.tensor
        the corresponding running mean
    """
    running_mean = torch.zeros(n_out)
    running_var = torch.ones(n_out)
    return running_var, running_mean


def init_FC(n_out, n_in):
    """Initialize a Fully Connected layer according to its size.

    Parameters
    ----------
    n_out : int
        the number of channels out
    n_in : int
        the number of channels int

    Returns
    -------
    weight : torch.tensor
        the initialized corresponding FC
    bias : torch.tensor
        the corresponding bias
    """
    weight = torch.empty(n_out, n_in)
    bias = torch.empty(n_out)

    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
    return weight, bias
