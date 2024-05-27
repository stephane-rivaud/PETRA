import pytest
import torch

from async_torch.layers.init_parameters import init_conv, init_FC, init_batchnorm
from async_torch.layers.layers import AsynchronousGenericLayer, AsynchronousFinal
from async_torch.optim.optimizer import add_optimizer
from async_torch.sequential_layers.sequential import SynchronousSequential, AsynchronousSequential

loss = torch.nn.functional.cross_entropy


class MiniConv(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(MiniConv, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_conv(n_out, n_in, 3)
        running_var, running_mean = init_batchnorm(n_out)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

        self._register_buffers('running_mean', running_mean)
        self._register_buffers('running_var', running_var)

    def local_f(self, x, weight_conv, bias_conv, running_mean, running_var, training):
        y = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=1, padding=1)
        x = torch.nn.functional.batch_norm(y, running_mean, running_var, weight=None, bias=None,
                                           training=training)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class MiniFinal(AsynchronousFinal):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(MiniFinal, self).__init__(loss, *args, **kwargs)
        self.size_averaging = None
        weight_conv, bias_conv = init_FC(n_out, n_in)
        self._register_parameters('weight', weight_conv)
        self._register_parameters('bias', bias_conv)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(x, y)

    def local_f(self, x, weight, bias, training):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.linear(x, weight, bias)
        return x


class MiniBottomBottleneck(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(MiniBottomBottleneck, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_conv(n_out, n_in, 3)
        running_var, running_mean = init_batchnorm(n_out)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

        self._register_buffers('running_mean', running_mean)
        self._register_buffers('running_var', running_var)

    def local_f(self, x, weight_conv, bias_conv, running_mean, running_var, training):
        y = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=1, padding=1)
        y = torch.nn.functional.batch_norm(y, running_mean, running_var, weight=None, bias=None,
                                           training=training)
        y = torch.nn.functional.relu(y, inplace=True)
        return x, y


class MiniTopBottleneck(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(MiniTopBottleneck, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_conv(n_out, n_in, 3)
        running_var, running_mean = init_batchnorm(n_out)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

        self._register_buffers('running_mean', running_mean)
        self._register_buffers('running_var', running_var)

    def local_f(self, x, y, weight_conv, bias_conv, running_mean, running_var, training):
        z = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=1, padding=1)
        z = torch.nn.functional.batch_norm(z, running_mean, running_var, weight=None, bias=None,
                                           training=training)
        z = torch.nn.functional.relu(z, inplace=True)
        return x + z


class TestModelsSaving:
    def test_save_and_load(self):
        n_in, n_class, bs = 3, 5, 7
        hidden_sizes = torch.randint(3, 7, (2,))  # generate a random number of channels for layers
        arch_1 = [MiniConv(n_in, hidden_sizes[0]),
                  MiniConv(hidden_sizes[0], hidden_sizes[1]),
                  MiniFinal(hidden_sizes[1], n_class)]

        arch_2 = [MiniConv(n_in, hidden_sizes[0]),
                  MiniConv(hidden_sizes[0], hidden_sizes[1]),
                  MiniFinal(hidden_sizes[1], n_class)]

        net_1 = SynchronousSequential(arch_1)
        net_2 = SynchronousSequential(arch_2)

        input_size = 20
        x = torch.randn(bs, n_in, input_size, input_size)

        _, y_1 = net_1.forward(x, 0)
        _, y_2 = net_2.forward(x, 0)
        assert not torch.allclose(y_1, y_2)

        state = net_1.state_list()
        net_2.load_state_list(state)

        _, y_1 = net_1.forward(x, 0)
        _, y_2 = net_2.forward(x, 0)
        assert torch.allclose(y_1, y_2)


class TestModelsCreation:
    @pytest.mark.parametrize('n_in', [2, 3])
    @pytest.mark.parametrize('n_class', [7, 10])
    def test_two_layers_nn_sync(self, n_in, n_class):
        hidden_sizes = torch.randint(3, 7, (2,))  # generate a random number of channels for layers
        arch = [MiniConv(n_in, hidden_sizes[0]),
                MiniConv(hidden_sizes[0], hidden_sizes[1]),
                MiniFinal(hidden_sizes[1], n_class)]
        for layer in arch:
            add_optimizer(layer, 'sgd', {'lr': 0.1})

        net = SynchronousSequential(arch)

        bs = 15
        input_size = 20
        x = torch.randn(bs, n_in, input_size, input_size)
        y_label = torch.randint(0, n_class, (bs,))

        _, y = net.forward(x, 0)  # train mode
        assert y.shape == (bs, n_class)
        for module in net.modules:
            _ = module.buffer.get()
            assert len(module.buffer) == 0  # empty buffers

        loss_, pred_, idx_ = net.forward_and_update(x, y_label, 0)
        assert torch.allclose(loss_, loss(pred_, y_label))

        # no reason prediction and loss should stay the same
        loss__, pred__, idx__ = net.forward_and_update(x, y_label, 1)

        assert not torch.allclose(pred__, pred_)
        assert not torch.allclose(loss__, loss_)
        assert torch.allclose(loss__, loss(pred__, y_label))

    @pytest.mark.parametrize('n_in', [2, 3])
    @pytest.mark.parametrize('n_class', [7, 10])
    def test_two_layers_nn_async(self, n_in, n_class):
        hidden_sizes = torch.randint(3, 7, (2,))  # generate a random number of channels for layers
        arch = [MiniConv(n_in, hidden_sizes[0]),
                MiniConv(hidden_sizes[0], hidden_sizes[1]),
                MiniFinal(hidden_sizes[1], n_class)]
        for layer in arch:
            add_optimizer(layer, 'sgd', {'lr': 0.1})
        net = AsynchronousSequential(arch)
        depth = len(net.modules)

        bs = 15
        input_size = 20
        x = torch.randn(bs, n_in, input_size, input_size)
        y_label = torch.randint(0, n_class, (bs,))

        _, y = net.forward(x, 0)  # train mode
        assert y.shape == (bs, n_class)
        for module in net.modules:
            _ = module.buffer.get()
            assert len(module.buffer) == 0  # empty buffers

        # exactly 'depth' works
        loss_, pred_, y_ = None, None, None
        for i in range(depth):
            loss_, pred_, y_ = net.forward_and_update(x, y_label, 0)

        assert torch.allclose(loss_, loss(pred_, y_label))

        # no reason loss should stay the same
        loss__, pred__, y__ = net.forward_and_update(x, y_label, 1)
        assert not torch.allclose(loss__, loss_)
        assert torch.allclose(loss__, loss(pred__, y_label))
