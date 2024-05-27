import torch

from ..layers.init_parameters import init_conv, init_batchnorm, init_FC
from ..layers.layers import AsynchronousGenericLayer, AsynchronousFinal
from ..layers.compression import QuantizSimple


class ConvBNReLU(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, eps_bn=1e-05,
                 momentum_bn=0.1, *args, **kwargs):
        super(ConvBNReLU, self).__init__(*args, **kwargs)
        self.stride = stride
        self.padding = padding
        self.momentum_bn = momentum_bn
        self.eps_bn = eps_bn

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
        x = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=self.stride, padding=self.padding)
        x = torch.nn.functional.batch_norm(x, running_mean, running_var, weight=weight_bn, bias=bias_bn,
                                           training=training, momentum=self.momentum_bn, eps=self.eps_bn)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class FlattenFullyConnectedCE(AsynchronousFinal):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(FlattenFullyConnectedCE, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_FC(n_out, n_in)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)

    def local_f(self, x, weight, bias, training):
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.linear(x, weight, bias)
        return x


def make_layers_VGG(cfg=(64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'), nclass=10,
                    store_param=True, store_vjp=False, quantizer=QuantizSimple, accumulation_steps=1,
                    accumulation_averaging=False):
    layers = []
    in_channels = 3
    depth = 0
    s = 1
    first_layer = True

    for x in cfg:
        if x == 'A':
            s = 2
        else:
            if s == 1 and x == 512:
                s = 2
            layers += [ConvBNReLU(in_channels, x, kernel_size=3, padding=1, stride=s, first_layer=first_layer,
                                  store_param=store_param, store_vjp=store_vjp, quantizer=quantizer,
                                  accumulation_steps=accumulation_steps, accumulation_averaging=accumulation_averaging)]
            in_channels = x
            s = 1
        depth += 1
        first_layer = False
    layers += [FlattenFullyConnectedCE(512, nclass, quantizer=quantizer, accumulation_steps=accumulation_steps,
                                       accumulation_averaging=accumulation_averaging)]

    return layers
