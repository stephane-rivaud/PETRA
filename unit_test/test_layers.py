import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)  # issue for conv2d

from async_torch.layers.layers import AsynchronousGenericLayer
from async_torch.optim.optimizer import add_optimizer


class ConvBNReLU(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, 3)
        self.BN = nn.BatchNorm2d(n_out)

    def forward(self, x):
        x = F.relu(self.BN(self.conv(x)))
        return x


class TestSumVariable(AsynchronousGenericLayer):
    def __init__(self, *args, **kwargs):
        super(TestSumVariable, self).__init__(*args, **kwargs)
        pass

    def local_f(self, x, training):
        return x + 2


class TestSumVariableLearned(AsynchronousGenericLayer):
    def __init__(self, a, b, *args, **kwargs):
        super(TestSumVariableLearned, self).__init__(*args, **kwargs)
        self._register_parameters('a', a)
        self._register_parameters('b', b)

    def local_f(self, x, a, b, training):
        return x + a


class TestConvolution(AsynchronousGenericLayer):
    def __init__(self, weight_conv, bias_conv, weight_bn, bias_bn, running_mean, running_var, stride, padding,
                 momentum_bn, eps_bn, *args, **kwargs):
        super(TestConvolution, self).__init__(*args, **kwargs)
        self.stride = stride
        self.padding = padding
        self.momentum_bn = momentum_bn
        self.eps_bn = eps_bn
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
        return x


def test_function_without_parameters():
    module = TestSumVariable()
    x = torch.zeros(1)
    module.forward(x)


def test_function_with_Parameters():
    a = torch.zeros(1)
    b = torch.zeros(1)
    module = TestSumVariableLearned(a, b)
    x = torch.zeros(1)
    module.forward(x, 1)
    module.backward(None, x, 1)


def test_function_with_parameters_To():
    a = torch.zeros(1)
    b = torch.zeros(1)
    module = TestSumVariableLearned(a, b)
    module.to(torch.float16)
    assert module.a_backward.dtype == torch.float16


@pytest.mark.parametrize("optimizer_class", ['sgd', 'adam'])
@pytest.mark.parametrize("lr", [0, 0.1])
def test_convolution_forward(optimizer_class, lr):
    # models
    model = ConvBNReLU(3, 7)
    conv = model.conv
    bn = model.BN
    conv2 = TestConvolution(conv.weight.clone(), conv.bias.clone(),
                            bn.weight.clone(), bn.bias.clone(),
                            bn.running_mean.clone(), bn.running_var.clone(),
                            1, 0, 0.1, 1e-05)

    # optimizers
    if optimizer_class == 'sgd':
        optim_kwargs = dict(
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            maximize=False
            )
        optimizer = torch.optim.SGD(model.parameters(), **optim_kwargs)
        add_optimizer(conv2, optimizer_class, optim_kwargs)
    elif optimizer_class == 'lars':
        raise ValueError(f'Test not implemented for this optimizer class ({optimizer_class}).')
    elif optimizer_class == 'adam':
        optim_kwargs = dict(
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=5e-4,
            amsgrad=False,
            maximize=False
            )
        optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
        beta1, beta2 = optim_kwargs.pop('betas')
        optim_kwargs['beta1'] = beta1
        optim_kwargs['beta2'] = beta2
        add_optimizer(conv2, optimizer_class, optim_kwargs)
    else:
        raise ValueError(f'Wrong optimizer class ({optimizer_class})')

    optimizer.zero_grad()
    # test forward
    x_ = torch.randn(127, 3, 45, 76)
    x = torch.nn.parameter.Parameter(x_.clone())
    y = model.forward(x)
    y_ = conv2.forward(x_, 1)
    assert torch.allclose(y, y_)

    # test backward
    e = torch.randn(127, 7, 43, 74)
    y.backward(e)
    g = x.grad.data
    _, g_, _ = conv2.backward(None, e, 1)
    assert torch.allclose(g, g_)

    x = torch.randn(127, 3, 45, 76)
    x_ = x.clone()
    model.eval()
    y = model.forward(x)
    y_ = conv2.forward(x_)

    assert torch.allclose(y, y_)
    assert torch.allclose(conv.weight, conv2.weight_conv_backward)

    # test update
    optimizer.step()
    conv2.update()
    optimizer.zero_grad()
    assert torch.allclose(conv.weight, conv2.weight_conv_backward)
    assert torch.allclose(conv.bias, conv2.bias_conv_backward)
    assert torch.allclose(bn.weight, conv2.weight_bn_backward)
    assert torch.allclose(bn.bias, conv2.bias_bn_backward)

    conv2.synchronize_forwardbackward()
    x_ = torch.randn(127, 3, 45, 76)
    x = torch.nn.parameter.Parameter(x_.clone())
    model.eval()
    y = model.forward(x)
    y_ = conv2.forward(x_)

    assert torch.allclose(y, y_)

    # Now, we check that parameters are not changed if there was no update
    optimizer.step()
    conv2.update()
    optimizer.zero_grad()
    model.train()

    assert torch.allclose(conv.weight, conv2.weight_conv_backward)

    conv2.synchronize_forwardbackward()

    # Now, we check it accumulates
    for i in range(3):
        # test backward
        x_ = torch.randn(127, 3, 45, 76)
        x = x_.clone()

        e = torch.randn(127, 7, 43, 74)
        y = model.forward(x)
        y.backward(e)
        y_ = conv2.forward(x_, i)
        _ = conv2.backward(None, e, i)
        assert torch.allclose(conv.weight.grad.data, conv2.weight_conv_grad_backward)

    # test update
    optimizer.step()
    conv2.update()
    optimizer.zero_grad()

    conv2.synchronize_forwardbackward()

    model.eval()
    assert torch.allclose(conv.weight, conv2.weight_conv_backward)

    # let's verify it works with test:
    x = torch.randn(127, 3, 45, 76)
    x_ = x.clone()
    y = model.forward(x)
    y_ = conv2.forward(x_)
    assert torch.allclose(y, y_)
