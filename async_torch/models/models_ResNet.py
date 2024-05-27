import math

import torch

from async_torch.layers.compression import QuantizSimple
from async_torch.layers.init_parameters import init_conv, init_batchnorm, init_FC
from async_torch.layers.layers import AsynchronousGenericLayer, AsynchronousFinal


class AVGFlattenFullyConnectedCE(AsynchronousFinal):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(AVGFlattenFullyConnectedCE, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_FC(n_out, n_in)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)

    def local_f(self, x, weight, bias, training):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.linear(x, weight, bias)
        return x


class ConvBNReLUMax(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, eps_bn=1e-05,
                 momentum_bn=0.1, max_pool=True, *args, **kwargs):
        super(ConvBNReLUMax, self).__init__(*args, **kwargs)
        self.stride = stride
        self.padding = padding
        self.momentum_bn = momentum_bn
        self.eps_bn = eps_bn
        self.max_pool = max_pool

        weight_conv, bias_conv = init_conv(n_out, n_in, kernel_size)
        weight_bn, bias_bn = init_batchnorm(n_out)
        running_var, running_mean = init_batchnorm(n_out)

        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)
        self._register_parameters('weight_bn', weight_bn)
        self._register_parameters('bias_bn', bias_bn)
        self._register_buffers('running_mean', running_mean)
        self._register_buffers('running_var', running_var)

    def local_f(self, x, weight_conv, bias_conv, weight_bn, bias_bn, running_mean, running_var, training):
        y = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=self.stride, padding=self.padding)
        x = torch.nn.functional.batch_norm(y, running_mean, running_var, weight=weight_bn, bias=bias_bn,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)
        x = torch.nn.functional.relu(x, inplace=True)
        if self.max_pool:
            x = torch.nn.functional.max_pool2d(x, 2)
        return x


class BasicBlock(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, downsample=False, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 eps_bn=1e-05, momentum_bn=0.1, max_pool=True, *args, **kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)
        self.stride = stride
        self.padding = padding
        self.momentum_bn = momentum_bn
        self.eps_bn = eps_bn
        self.max_pool = max_pool
        self.downsample = downsample

        weight_conv_1, bias_conv_1 = init_conv(n_out, n_in, kernel_size)
        weight_bn_1, bias_bn_1 = init_batchnorm(n_out)
        running_var_1, running_mean_1 = init_batchnorm(n_out)

        self._register_parameters('weight_conv_1', weight_conv_1)
        self._register_parameters('bias_conv_1', bias_conv_1)
        self._register_parameters('weight_bn_1', weight_bn_1)
        self._register_parameters('bias_bn_1', bias_bn_1)
        self._register_buffers('running_mean', running_mean_1)
        self._register_buffers('running_var_1', running_var_1)

        if downsample:
            weight_conv_ds, bias_conv_ds = init_conv(n_out, n_in, 1)
            weight_bn_ds, bias_bn_ds = init_batchnorm(n_out)
            running_var_ds, running_mean_ds = init_batchnorm(n_out)
        else:
            weight_conv_ds, bias_conv_ds = torch.empty(0), torch.empty(0)
            weight_bn_ds, bias_bn_ds = torch.empty(0), torch.empty(0)
            running_var_ds, running_mean_ds = torch.empty(0), torch.empty(0)

        self._register_parameters('weight_conv_ds', weight_conv_ds)
        self._register_parameters('bias_conv_ds', bias_conv_ds)
        self._register_parameters('weight_bn_ds', weight_bn_ds)
        self._register_parameters('bias_bn_ds', bias_bn_ds)
        self._register_buffers('running_mean_ds', running_mean_ds)
        self._register_buffers('running_var_ds', running_var_ds)

        weight_conv_2, bias_conv_2 = init_conv(n_out, n_out, kernel_size)
        weight_bn_2, bias_bn_2 = init_batchnorm(n_out)
        running_var_2, running_mean_2 = init_batchnorm(n_out)

        self._register_parameters('weight_conv_2', weight_conv_2)
        self._register_parameters('bias_conv_2', bias_conv_2)
        self._register_parameters('weight_bn_2', weight_bn_2)
        self._register_parameters('bias_bn_2', bias_bn_2)
        self._register_buffers('running_mean_2', running_mean_2)
        self._register_buffers('running_var_2', running_var_2)

    def local_f(self, x, weight_conv_1, bias_conv_1, weight_bn_1, bias_bn_1,
                weight_conv_ds, bias_conv_ds, weight_bn_ds, bias_bn_ds,
                weight_conv_2, bias_conv_2, weight_bn_2, bias_bn_2, running_mean_1, running_var_1,
                running_mean_ds, running_var_ds, running_mean_2, running_var_2, training):
        y = torch.nn.functional.conv2d(x, weight_conv_1, bias=bias_conv_1, stride=self.stride, padding=self.padding)
        y = torch.nn.functional.batch_norm(y, running_mean_1, running_var_1, weight=weight_bn_1, bias=bias_bn_1,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)
        y = torch.nn.functional.relu(y, inplace=True)
        if self.downsample:
            x_ds = torch.nn.functional.conv2d(x, weight_conv_ds, bias=bias_conv_ds, stride=2, padding=0)
            x_ds = torch.nn.functional.batch_norm(x_ds, running_mean_ds, running_var_ds, weight=weight_bn_ds,
                                                  bias=bias_bn_ds, training=training, momentum=self.momentum_bn,
                                                  eps=self.eps_bn)
        else:
            x_ds = x

        y = torch.nn.functional.conv2d(y, weight_conv_2, bias=bias_conv_2, stride=1, padding=self.padding)
        y = torch.nn.functional.batch_norm(y, running_mean_2, running_var_2, weight=weight_bn_2, bias=bias_bn_2,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)
        y += x_ds
        y = torch.nn.functional.relu(y, inplace=True)
        return y


class Bottleneck(AsynchronousGenericLayer):
    def __init__(self, n_in, n_hidden, n_out, downsample=False, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, eps_bn=1e-05,
                 momentum_bn=0.1, max_pool=True, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)
        self.stride = stride
        self.padding = padding
        self.momentum_bn = momentum_bn
        self.eps_bn = eps_bn
        self.max_pool = max_pool
        self.downsample = downsample

        weight_conv_1, bias_conv_1 = init_conv(n_hidden, n_in, 1)
        weight_bn_1, bias_bn_1 = init_batchnorm(n_hidden)
        running_var_1, running_mean_1 = init_batchnorm(n_hidden)

        self._register_parameters('weight_conv_1', weight_conv_1)
        self._register_parameters('bias_conv_1', bias_conv_1)
        self._register_parameters('weight_bn_1', weight_bn_1)
        self._register_parameters('bias_bn_1', bias_bn_1)
        self._register_buffers('running_mean', running_mean_1)
        self._register_buffers('running_var_1', running_var_1)

        weight_conv_2, bias_conv_2 = init_conv(n_hidden, n_hidden, kernel_size)
        weight_bn_2, bias_bn_2 = init_batchnorm(n_hidden)
        running_var_2, running_mean_2 = init_batchnorm(n_hidden)

        self._register_parameters('weight_conv_2', weight_conv_2)
        self._register_parameters('bias_conv_2', bias_conv_2)
        self._register_parameters('weight_bn_2', weight_bn_2)
        self._register_parameters('bias_bn_2', bias_bn_2)
        self._register_buffers('running_mean_2', running_mean_2)
        self._register_buffers('running_var_2', running_var_2)

        weight_conv_3, bias_conv_3 = init_conv(n_out, n_hidden, 1)
        weight_bn_3, bias_bn_3 = init_batchnorm(n_out)
        running_var_3, running_mean_3 = init_batchnorm(n_out)

        self._register_parameters('weight_conv_3', weight_conv_3)
        self._register_parameters('bias_conv_3', bias_conv_3)
        self._register_parameters('weight_bn_3', weight_bn_3)
        self._register_parameters('bias_bn_3', bias_bn_3)
        self._register_buffers('running_mean_3', running_mean_3)
        self._register_buffers('running_var_3', running_var_3)

        if downsample:
            weight_conv_ds, bias_conv_ds = init_conv(n_out, n_in, 1)
            weight_bn_ds, bias_bn_ds = init_batchnorm(n_out)
            running_var_ds, running_mean_ds = init_batchnorm(n_out)
        else:
            weight_conv_ds, bias_conv_ds = torch.empty(0), torch.empty(0)
            weight_bn_ds, bias_bn_ds = torch.empty(0), torch.empty(0)
            running_var_ds, running_mean_ds = torch.empty(0), torch.empty(0)

        self._register_parameters('weight_conv_ds', weight_conv_ds)
        self._register_parameters('bias_conv_ds', bias_conv_ds)
        self._register_parameters('weight_bn_ds', weight_bn_ds)
        self._register_parameters('bias_bn_ds', bias_bn_ds)
        self._register_buffers('running_mean_ds', running_mean_ds)
        self._register_buffers('running_var_ds', running_var_ds)

    def local_f(self, x,
                weight_conv_1, bias_conv_1, weight_bn_1, bias_bn_1,
                weight_conv_2, bias_conv_2, weight_bn_2, bias_bn_2,
                weight_conv_3, bias_conv_3, weight_bn_3, bias_bn_3,
                weight_conv_ds, bias_conv_ds, weight_bn_ds, bias_bn_ds,
                running_mean_1, running_var_1,
                running_mean_2, running_var_2,
                running_mean_3, running_var_3,
                running_mean_ds, running_var_ds,
                training):
        y = torch.nn.functional.conv2d(x, weight_conv_1, bias=bias_conv_1, stride=self.stride, padding=0)
        y = torch.nn.functional.batch_norm(y, running_mean_1, running_var_1, weight=weight_bn_1, bias=bias_bn_1,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)
        y = torch.nn.functional.relu(y, inplace=True)
        if self.downsample:
            x_ds = torch.nn.functional.conv2d(x, weight_conv_ds, bias=bias_conv_ds, stride=self.stride, padding=0)
            x_ds = torch.nn.functional.batch_norm(x_ds, running_mean_ds, running_var_ds, weight=weight_bn_ds,
                                                  bias=bias_bn_ds, training=training, momentum=self.momentum_bn,
                                                  eps=self.eps_bn)
        else:
            x_ds = x

        y = torch.nn.functional.conv2d(y, weight_conv_2, bias=bias_conv_2, stride=1, padding=self.padding)
        y = torch.nn.functional.batch_norm(y, running_mean_2, running_var_2, weight=weight_bn_2, bias=bias_bn_2,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)

        y = torch.nn.functional.conv2d(y, weight_conv_3, bias=bias_conv_3, stride=1, padding=0)
        y = torch.nn.functional.batch_norm(y, running_mean_3, running_var_3, weight=weight_bn_3, bias=bias_bn_3,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)

        y += x_ds
        y = torch.nn.functional.relu(y, inplace=True)
        return y


def make_layers_resnet18(dataset, nclass=10, last_bn_zero_init=False, store_param=True, store_vjp=False,
                         quantizer=QuantizSimple, accumulation_steps=1, accumulation_averaging=False):
    layers = []
    in_channels = 3
    inplanes = 64
    channels = [64, 128, 256, 512]
    if dataset == 'imagenet':
        kernel_size, stride, padding, max_pool = 7, 2, 3, True
    else:
        kernel_size, stride, padding, max_pool = 3, 1, 1, False

    layers += [ConvBNReLUMax(in_channels, channels[0], kernel_size=kernel_size, padding=padding, stride=stride,
                             max_pool=max_pool, first_layer=True, store_param=store_param, store_vjp=store_vjp,
                             quantizer=quantizer, accumulation_steps=accumulation_steps,
                             accumulation_averaging=accumulation_averaging)]
    x = inplanes
    for c in channels:
        if x != c:
            layers += [
                BasicBlock(x, c, stride=2, downsample=True, padding=1, store_param=store_param, store_vjp=store_vjp,
                           quantizer=quantizer, accumulation_steps=accumulation_steps,
                           accumulation_averaging=accumulation_averaging)]
        else:
            layers += [BasicBlock(x, c, padding=1, store_param=store_param, store_vjp=store_vjp, quantizer=quantizer,
                                  accumulation_steps=accumulation_steps, accumulation_averaging=accumulation_averaging)]
        layers += [BasicBlock(c, c, padding=1, store_param=store_param, store_vjp=store_vjp, quantizer=quantizer,
                              accumulation_steps=accumulation_steps, accumulation_averaging=accumulation_averaging)]
        x = c

    if last_bn_zero_init:
        for layer in layers[1:]:
            name = 'weight_bn_2'
            setattr(layer, name + '_forward', torch.zeros_like(getattr(layer, name + '_forward')))
            setattr(layer, name + '_backward', torch.zeros_like(getattr(layer, name + '_backward')))

    # Need avg pooling
    layers += [AVGFlattenFullyConnectedCE(512, nclass, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                          accumulation_averaging=accumulation_averaging)]
    return layers


def resnet18_2_memory(dataset, nclass=10, store_param=True, batch_size=128, accumulation_steps=1):
    layers = make_layers_resnet18(dataset, nclass, store_param=store_param)
    if dataset == 'imagenet':
        input_sizes = [
            (3, 224, 224),
            (64, 56, 56),
            (64, 56, 56),
            (64, 56, 56),
            (128, 28, 28),
            (128, 28, 28),
            (256, 14, 14),
            (256, 14, 14),
            (512, 7, 7),
            (512, 7, 7),
        ]
    else:
        input_sizes = [
            (3, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (128, 16, 16),
            (128, 16, 16),
            (256, 8, 8),
            (256, 8, 8),
            (512, 4, 4),
            (512, 4, 4),
        ]
    compute_memory('resnet18', layers, input_sizes, batch_size, accumulation_steps, store_param)


def make_layers_resnet34(dataset, nclass=10, store_param=True, store_vjp=False, quantizer=QuantizSimple,
                         accumulation_steps=1, accumulation_averaging=False):
    layers = []
    in_channels = 3
    inplanes = 64
    channels = [64, 64, 64,
                128, 128, 128, 128,
                256, 256, 256, 256, 256, 256,
                512, 512, 512]
    if dataset == 'imagenet':
        kernel_size, stride, padding, max_pool = 7, 2, 3, True
    else:
        kernel_size, stride, padding, max_pool = 3, 1, 1, False
    layers += [ConvBNReLUMax(in_channels, channels[0], kernel_size=kernel_size, padding=padding, stride=stride,
                             max_pool=max_pool, first_layer=True, store_param=store_param, store_vjp=store_vjp,
                             quantizer=quantizer, accumulation_steps=accumulation_steps,
                             accumulation_averaging=accumulation_averaging)]

    x = inplanes
    for c in channels:
        if x != c:
            layers += [
                BasicBlock(x, c, stride=2, downsample=True, padding=1, store_param=store_param, store_vjp=store_vjp,
                           quantizer=quantizer, accumulation_steps=accumulation_steps,
                           accumulation_averaging=accumulation_averaging)]
        else:
            layers += [BasicBlock(x, c, padding=1, store_param=store_param, store_vjp=store_vjp, quantizer=quantizer,
                                  accumulation_steps=accumulation_steps, accumulation_averaging=accumulation_averaging)]
        x = c

    # Need avg pooling
    layers += [AVGFlattenFullyConnectedCE(512, nclass, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                          accumulation_averaging=accumulation_averaging)]
    return layers


def resnet34_2_memory(dataset, nclass=10, store_param=True, batch_size=128, accumulation_steps=1):
    layers = make_layers_resnet34(dataset, nclass, store_param=store_param)
    if dataset == 'imagenet':
        input_sizes = [
            (3, 224, 224),
            (64, 56, 56),
            (64, 56, 56),
            (64, 56, 56),
            (64, 56, 56),
            (128, 28, 28),
            (128, 28, 28),
            (128, 28, 28),
            (128, 28, 28),
            (256, 14, 14),
            (256, 14, 14),
            (256, 14, 14),
            (256, 14, 14),
            (256, 14, 14),
            (256, 14, 14),
            (512, 7, 7),
            (512, 7, 7),
            (512, 7, 7),
        ]
    else:
        input_sizes = [
            (3, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (64, 32, 32),
            (128, 16, 16),
            (128, 16, 16),
            (128, 16, 16),
            (128, 16, 16),
            (256, 8, 8),
            (256, 8, 8),
            (256, 8, 8),
            (256, 8, 8),
            (256, 8, 8),
            (256, 8, 8),
            (512, 4, 4),
            (512, 4, 4),
            (512, 4, 4),
        ]
    compute_memory('resnet34', layers, input_sizes, batch_size, accumulation_steps, store_param)


def make_layers_resnet50(dataset, nclass=10, store_param=True, store_vjp=False, quantizer=QuantizSimple,
                         accumulation_steps=1, accumulation_averaging=False):
    layers = []
    in_channels = 3
    channels = [64,
                256, 256, 256,
                512, 512, 512, 512,
                1024, 1024, 1024, 1024, 1024, 1024,
                2048, 2048, 2048]
    hidden_sizes = [64, 64, 64,
                    128, 128, 128, 128,
                    256, 256, 256, 256, 256, 256,
                    512, 512, 512]

    if dataset == 'imagenet':
        kernel_size, stride, padding, max_pool = 7, 2, 3, True
    else:
        kernel_size, stride, padding, max_pool = 3, 1, 1, False

    layers += [ConvBNReLUMax(in_channels, channels[0], kernel_size=kernel_size, padding=padding, stride=stride,
                             max_pool=max_pool, first_layer=True, store_param=store_param, store_vjp=store_vjp,
                             quantizer=quantizer, accumulation_steps=accumulation_steps,
                             accumulation_averaging=accumulation_averaging)]

    for k, (n_in, n_h, n_out) in enumerate(zip(channels[:-1], hidden_sizes, channels[1:])):
        if k == 0:
            layers += [Bottleneck(n_in, n_h, n_out, stride=1, downsample=True, padding=1, store_param=store_param,
                                  store_vjp=store_vjp, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]
        elif n_in != n_out:
            layers += [Bottleneck(n_in, n_h, n_out, stride=2, downsample=True, padding=1, store_param=store_param,
                                  store_vjp=store_vjp, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]
        else:
            layers += [Bottleneck(n_in, n_h, n_out, padding=1, store_param=store_param, store_vjp=store_vjp,
                                  quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]

    # Need avg pooling
    layers += [
        AVGFlattenFullyConnectedCE(channels[-1], nclass, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                   accumulation_averaging=accumulation_averaging)]
    return layers


def resnet50_2_memory(dataset, nclass=10, store_param=True, batch_size=128, accumulation_steps=1):
    layers = make_layers_resnet50(dataset, nclass, store_param=store_param)
    if dataset == 'imagenet':
        input_sizes = [
            (3, 224, 224),
            (64, 56, 56), (256, 56, 56), (256, 56, 56),
            (256, 56, 56), (512, 28, 28), (512, 28, 28), (512, 28, 28),
            (512, 28, 28), (1024, 14, 14), (1024, 14, 14), (1024, 14, 14), (1024, 14, 14), (1024, 14, 14),
            (1024, 14, 14), (2048, 7, 7), (2048, 7, 7),
            (2048, 7, 7),
        ]
    else:
        input_sizes = [
            (3, 32, 32),
            (64, 32, 32), (256, 32, 32), (256, 32, 32),
            (256, 32, 32), (512, 16, 16), (512, 16, 16), (512, 16, 16),
            (512, 16, 16), (1024, 8, 8), (1024, 8, 8), (1024, 8, 8), (1024, 8, 8), (1024, 8, 8),
            (1024, 8, 8), (2048, 4, 4), (2048, 4, 4),
            (2048, 4, 4)
        ]
    compute_memory('resnet50', layers, input_sizes, batch_size, accumulation_steps, store_param)


def make_layers_resnet101_2(dataset, nclass=10, store_param=True, store_vjp=False, quantizer=QuantizSimple,
                            accumulation_steps=1, accumulation_averaging=False):
    layers = []
    in_channels = 3
    channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3
    hidden_sizes = [64] * 3 + [128] * 4 + [256] * 23 + [512] * 3
    if dataset == 'imagenet':
        kernel_size, stride, padding, max_pool = 7, 2, 3, True
    else:
        kernel_size, stride, padding, max_pool = 3, 1, 1, False

    layers += [ConvBNReLUMax(in_channels, channels[0], kernel_size=kernel_size, padding=padding, stride=stride,
                             max_pool=max_pool, first_layer=True, store_param=store_param, store_vjp=store_vjp,
                             quantizer=quantizer, accumulation_steps=accumulation_steps,
                             accumulation_averaging=accumulation_averaging)]

    for k, (n_in, n_h, n_out) in enumerate(zip(channels[:-1], hidden_sizes, channels[1:])):
        if k == 0:
            layers += [Bottleneck(n_in, n_h, n_out, stride=1, downsample=True, padding=1, store_param=store_param,
                                  store_vjp=store_vjp, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]
        elif n_in != n_out:
            layers += [Bottleneck(n_in, n_h, n_out, stride=2, downsample=True, padding=1, store_param=store_param,
                                  store_vjp=store_vjp, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]
        else:
            layers += [Bottleneck(n_in, n_h, n_out, padding=1, store_param=store_param, store_vjp=store_vjp,
                                  quantizer=quantizer, accumulation_steps=accumulation_steps,
                                  accumulation_averaging=accumulation_averaging)]

    # Need avg pooling
    layers += [
        AVGFlattenFullyConnectedCE(channels[-1], nclass, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                   accumulation_averaging=accumulation_averaging)]
    return layers


def compute_memory(model, layers, input_sizes, batch_size, accumulation_steps, store_param):
    num_param = 0
    for layer in layers:
        for name in layer.list_parameters:
            num_param += layer.get_parameter(name).numel()
    print('-' * 50)
    print(f'Model: {model} - {num_param:,} params | Store ctx: {store_param} | Accumulation steps: {accumulation_steps}')
    print('-' * 50)
    depth = len(layers)
    staleness = [2 * (depth - j) for j in range(1, depth + 1)]
    downsample = [hasattr(layer, 'downsample') and layer.downsample for layer in layers]

    model_memory = 0
    total_input_buffer = 0
    total_param_buffer = 0
    for k, layer in enumerate(layers):
        num_param = sum([layer.get_parameter(name).numel() for name in layer.list_parameters])
        num_buffer = sum([layer.get_buffer(name).numel() for name in layer.list_buffers])
        input_size = batch_size * input_sizes[k][0] * input_sizes[k][1] * input_sizes[k][2]

        # lower bound on the memory needed for the local computations
        layer_memory = num_param + num_buffer + input_size
        model_memory += layer_memory

        # size of the input buffer
        if k >= 0:
            input_buffer_size = input_size * staleness[k]
            layer_memory += input_buffer_size
            total_input_buffer += input_buffer_size

        # size of the param buffer
        if store_param:
            param_buffer_size = num_param * math.ceil(staleness[k] / accumulation_steps)
            total_param_buffer += param_buffer_size
            layer_memory += param_buffer_size

        # convert to Giga-bytes
        layer_memory = layer_memory * 32 / 10 ** 9 / 8
        message = f"Block {k}: {layer.__class__.__name__} | " \
                  f"{num_param:,} params | " \
                  f"{num_buffer:,} buffers | " \
                  f"input {input_sizes[k]} | " \
                  f"staleness {staleness[k]} | " \
                  f"downsample {downsample[k]} | " \
                  f"memory {layer_memory: .2f}GB"
        print(message)

    # total memory
    total_buffer = total_input_buffer + total_param_buffer
    total_memory = total_buffer + model_memory

    # convert to Giga-bytes
    model_memory *= 32 / 10 ** 9 / 8
    total_input_buffer *= 32 / 10 ** 9 / 8
    total_param_buffer *= 32 / 10 ** 9 / 8
    total_buffer *= 32 / 10 ** 9 / 8
    total_memory *= 32 / 10 ** 9 / 8
    print(f"Model: {model_memory: .2f}GB -- Input: {total_input_buffer: .2f}GB -- Param: {total_param_buffer: .2f}GB -- Buffer: {total_buffer: .2f}GB -- Total: {total_memory: .2f}GB")


if __name__ == '__main__':
    dataset = 'cifar10'
    model = 'resnet50'
    n_class = 10 if dataset == 'cifar10' else 1000
    batch_size = 128 if dataset == 'cifar10' else 64

    if model == 'resnet18':
        memory_function = resnet18_2_memory
    elif model == 'resnet34':
        memory_function = resnet34_2_memory
    elif model == 'resnet50':
        memory_function = resnet50_2_memory
    else:
        raise ValueError(f'Wrong model ({model})')

    store_param = True
    accumulation_steps = 1
    memory_function(dataset, n_class, batch_size=batch_size, store_param=store_param,
                    accumulation_steps=accumulation_steps)
    print()
