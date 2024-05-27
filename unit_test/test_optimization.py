import pytest
import torch
import torch.nn as nn

from async_torch.layers.init_parameters import init_conv, init_FC, init_batchnorm
from async_torch.layers.layers import AsynchronousGenericLayer, AsynchronousFinal
from async_torch.optim.optimizer import add_optimizer
from async_torch.sequential_layers.sequential import SynchronousSequential, AsynchronousSequential

loss_fn = nn.functional.cross_entropy
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ConvBnReLU(AsynchronousGenericLayer):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(ConvBnReLU, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_conv(n_out, n_in, 3)
        weight_bn, bias_bn = init_batchnorm(n_out)
        running_var, running_mean = init_batchnorm(n_out)
        self._register_parameters('weight_conv', weight_conv)
        self._register_parameters('bias_conv', bias_conv)

        self._register_parameters('weight_bn', weight_bn)
        self._register_parameters('bias_bn', bias_bn)
        self._register_buffers('running_mean', running_mean)
        self._register_buffers('running_var', running_var)

    def local_f(self, x, weight_conv, bias_conv, weight_bn, bias_bn, running_mean, running_var, training):
        x = torch.nn.functional.conv2d(x, weight_conv, bias=bias_conv, stride=1, padding=1)
        x = torch.nn.functional.batch_norm(x, running_mean, running_var, weight=weight_bn, bias=bias_bn,
                                           training=training)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class LinearFinal(AsynchronousFinal):
    def __init__(self, n_in, n_out, *args, **kwargs):
        super(LinearFinal, self).__init__(*args, **kwargs)
        weight_conv, bias_conv = init_FC(n_out, n_in)
        self._register_parameters('weight', weight_conv)
        self._register_parameters('bias', bias_conv)

    def loss(self, x, y):
        return loss_fn(x, y)

    def local_f(self, x, weight, bias, training):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.linear(x, weight, bias)
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def compare_pytorch_and_async(tensor1, tensor2, tol=1e-5):
    l_infty = (tensor1 - tensor2).abs().max().item()
    pytorch_infty = tensor1.abs().max().item()
    async_infty = tensor2.abs().max().item()
    relative_infty_diff = 2 * l_infty / (pytorch_infty + async_infty)

    l2 = (tensor1 - tensor2).abs().mean().item()
    pytorch_l2 = tensor1.abs().mean().item()
    async_l2 = tensor2.abs().mean().item()
    relative_l2_diff = 2 * l2 / (pytorch_l2 + async_l2)

    print(
        f'L infinity difference: {l_infty: .4e} -- L inifinty norm: pytorch {pytorch_infty: .4e}, async {async_infty: .4e} -- relative difference: {relative_infty_diff: .4e}')
    print(
        f'L2 difference: {l2: .4e} -- L2 norm: pytorch {pytorch_l2: .4e}, async {async_l2: .4e} -- relative difference: {relative_l2_diff: .4e}')
    return torch.allclose(tensor1, tensor2, atol=tol)


def compare_tensors(tensor1, tensor2, preamble=None):
    if not torch.allclose(tensor1, tensor2):
        l_infty = (tensor1 - tensor2).abs().max().item()
        infty_norm_1 = tensor1.abs().max().item()
        infty_norm_2 = tensor2.abs().max().item()
        relative_infty_diff = 2 * l_infty / (infty_norm_1 + infty_norm_2)

        l2 = (tensor1 - tensor2).abs().mean().item()
        norm2_1 = tensor1.abs().mean().item()
        norm2_2 = tensor2.abs().mean().item()
        relative_l2_diff = 2 * l2 / (norm2_1 + norm2_2)

        if preamble is not None:
            print(preamble)
        print(
            f'L infinity difference: {l_infty: .4e} -- L inifinty norm: tensor1 {infty_norm_1: .4e}, tensor2 {infty_norm_2: .4e} -- relative difference: {relative_infty_diff: .4e}')
        print(
            f'L2 difference: {l2: .4e} -- L2 norm: tensor1 {norm2_1: .4e}, tensor2 {norm2_2: .4e} -- relative difference: {relative_l2_diff: .4e}')
    return torch.allclose(tensor1, tensor2)


class TestOptimization:
    @pytest.mark.parametrize('num_updates', [5])
    @pytest.mark.parametrize('store_vjp', [False, True])
    @pytest.mark.parametrize('no_bn_weight_decay', [False, True])
    @pytest.mark.parametrize('weight_decay', [0.0, 5e-1])
    @pytest.mark.parametrize('momentum', [0.0, 0.9])
    def test_synchronous_against_pytorch(self, momentum, weight_decay, no_bn_weight_decay, store_vjp, num_updates):
        """Test the optimization of a simple model with a single layer."""
        n_in, n_hidden, n_out = 3, 7, 10
        model_pytorch = nn.Sequential(
            nn.Conv2d(n_in, n_hidden, 3, padding=1),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(n_hidden, n_out)
        )

        # ----- Need to compare the updates performed by PyTorch against our custom optimizers.
        conv_bn_relu = ConvBnReLU(n_in, n_hidden, store_vjp=store_vjp)
        conv_bn_relu._register_parameters('weight_conv', model_pytorch[0].weight)
        conv_bn_relu._register_parameters('bias_conv', model_pytorch[0].bias)
        conv_bn_relu._register_parameters('weight_bn', model_pytorch[1].weight)
        conv_bn_relu._register_parameters('bias_bn', model_pytorch[1].bias)
        conv_bn_relu._register_buffers('running_mean', model_pytorch[1].running_mean)
        conv_bn_relu._register_buffers('running_var', model_pytorch[1].running_var)

        linear_final = LinearFinal(n_hidden, n_out)
        linear_final._register_parameters('weight', model_pytorch[5].weight)
        linear_final._register_parameters('bias', model_pytorch[5].bias)

        model_sync = SynchronousSequential([conv_bn_relu, linear_final])
        model_sync.synchronize_layers()

        assert torch.allclose(model_pytorch[0].weight,
                              conv_bn_relu.get_parameter('weight_conv', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[0].weight,
                                      conv_bn_relu.get_parameter('weight_conv', mode='forward'))
        assert torch.allclose(model_pytorch[0].bias, conv_bn_relu.get_parameter('bias_conv', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[0].bias,
                                      conv_bn_relu.get_parameter('bias_conv', mode='forward'))
        assert torch.allclose(model_pytorch[1].weight, conv_bn_relu.get_parameter('weight_bn', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[1].weight,
                                      conv_bn_relu.get_parameter('weight_bn', mode='forward'))
        assert torch.allclose(model_pytorch[1].bias, conv_bn_relu.get_parameter('bias_bn', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[1].bias,
                                      conv_bn_relu.get_parameter('bias_bn', mode='forward'))
        assert torch.allclose(model_pytorch[1].running_mean,
                              conv_bn_relu.get_buffer('running_mean', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[1].running_mean, conv_bn_relu.get_buffer('running_mean'))
        assert torch.allclose(model_pytorch[1].running_var,
                              conv_bn_relu.get_buffer('running_var', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[1].running_var, conv_bn_relu.get_buffer('running_var'))
        assert torch.allclose(model_pytorch[5].weight, linear_final.get_parameter('weight', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[5].weight,
                                      linear_final.get_parameter('weight', mode='forward'))
        assert torch.allclose(model_pytorch[5].bias, linear_final.get_parameter('bias', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[5].bias, linear_final.get_parameter('bias', mode='forward'))

        # ----- Define optimizer
        lr = 0.1

        # pytorch model
        if no_bn_weight_decay:
            parameters = add_weight_decay(model_pytorch, weight_decay)
            optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        else:
            parameters = model_pytorch.parameters()
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)

        # async model
        optim_kwargs = {
            'lr': lr,
            'momentum': momentum
        }

        if no_bn_weight_decay:
            condition = lambda name: 'bias' not in name and 'bn' not in name
            optim_kwargs['weight_decay'] = weight_decay
            model_sync.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))

            condition = lambda name: 'bias' in name or 'bn' in name
            optim_kwargs['weight_decay'] = 0.
            model_sync.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))
        else:
            optim_kwargs['weight_decay'] = weight_decay
            model_sync.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs))

        # ----- Compare the forward pass in eval mode
        idx = None
        bs = 8
        x = torch.rand(bs, n_in, 32, 32)
        y = torch.randint(0, n_out, (bs,))

        # pytorch model
        model_pytorch.eval()
        with torch.no_grad():
            pred_pytorch = model_pytorch(x)
            loss_pytorch = loss_fn(pred_pytorch, y)

        # async model
        with torch.no_grad():
            loss_async, pred_async = model_sync.forward(x, idx, targets=y)

        assert torch.allclose(pred_pytorch, pred_async), compare_pytorch_and_async(pred_pytorch, pred_async)
        assert torch.allclose(loss_pytorch, loss_async), compare_pytorch_and_async(loss_pytorch, loss_async)

        # ----- Compare the forward pass in train mode
        x = torch.rand(bs, n_in, 32, 32)
        y = torch.randint(0, n_out, (bs,))
        model_pytorch.train()

        # pytorch model
        model_pytorch.train()
        with torch.no_grad():
            pred_pytorch = model_pytorch(x)
            loss_pytorch = loss_fn(pred_pytorch, y)

        # async model
        idx = 0
        with torch.no_grad():
            loss_async, pred_async = model_sync.forward(x, idx, targets=y)

        assert torch.allclose(pred_pytorch, pred_async), compare_pytorch_and_async(pred_pytorch, pred_async)
        assert torch.allclose(loss_pytorch, loss_async), compare_pytorch_and_async(loss_pytorch, loss_async)

        assert torch.allclose(model_pytorch[1].running_mean,
                              conv_bn_relu.get_buffer('running_mean', mode='forward')), \
            compare_pytorch_and_async(model_pytorch[1].running_mean,
                                      conv_bn_relu.get_buffer('running_mean', mode='forward'))
        assert torch.allclose(model_pytorch[1].running_var, conv_bn_relu.get_buffer('running_var')), \
            compare_pytorch_and_async(model_pytorch[1].running_var,
                                      conv_bn_relu.get_buffer('running_var', mode='forward'))

        # synchronize running stats for batch for the next forward pass
        conv_bn_relu._register_buffers('running_mean',
                                       conv_bn_relu.get_buffer('running_mean', mode='forward'))
        conv_bn_relu._register_buffers('running_var',
                                       conv_bn_relu.get_buffer('running_var', mode='forward'))

        # empty the buffer
        _ = model_sync.modules[0].buffer.get()
        assert len(model_sync.modules[0].buffer) == 0

        # ----- Compare the gradients and updates
        for _ in range(num_updates):
            model_pytorch.train()
            optimizer.zero_grad()

            # ----- Compare forward and backward pass
            # pytorch model
            pred_pytorch = model_pytorch(x)
            loss_pytorch = loss_fn(pred_pytorch, y)
            loss_pytorch.backward()

            # async model
            x_tmp = model_sync.modules[0].forward(x, idx)
            _, grad_x, meta = model_sync.modules[1].forward_and_backward(x_tmp, y, idx)
            _, _, _ = model_sync.modules[0].backward(None, grad_x, idx)
            loss_async, idx_async, pred_async, y_async = meta

            assert idx == idx_async
            assert torch.allclose(pred_pytorch, pred_async), compare_pytorch_and_async(pred_pytorch, pred_async)
            assert torch.allclose(loss_pytorch, loss_async), compare_pytorch_and_async(loss_pytorch, loss_async)

            # checking the gradients
            assert torch.allclose(model_pytorch[0].weight.grad, conv_bn_relu.get_gradient('weight_conv')), \
                compare_pytorch_and_async(model_pytorch[0].weight.grad, conv_bn_relu.get_gradient('weight_conv'))
            assert torch.allclose(model_pytorch[0].bias.grad, conv_bn_relu.get_gradient('bias_conv')), \
                compare_pytorch_and_async(model_pytorch[0].bias.grad, conv_bn_relu.get_gradient('bias_conv'))
            assert torch.allclose(model_pytorch[1].weight.grad, conv_bn_relu.get_gradient('weight_bn')), \
                compare_pytorch_and_async(model_pytorch[1].weight.grad, conv_bn_relu.get_gradient('weight_bn'))
            assert torch.allclose(model_pytorch[1].bias.grad, conv_bn_relu.get_gradient('bias_bn')), \
                compare_pytorch_and_async(model_pytorch[1].bias.grad, conv_bn_relu.get_gradient('bias_bn'))
            assert torch.allclose(model_pytorch[5].weight.grad, linear_final.get_gradient('weight')), \
                compare_pytorch_and_async(model_pytorch[5].weight.grad, linear_final.get_gradient('weight'))
            assert torch.allclose(model_pytorch[5].bias.grad, linear_final.get_gradient('bias')), \
                compare_pytorch_and_async(model_pytorch[5].bias.grad, linear_final.get_gradient('bias'))

            # ----- Compare the updates
            # pytorch
            optimizer.step()

            # async
            for module in model_sync.modules:
                module.update()
            model_sync.synchronize_layers()

            assert torch.allclose(model_pytorch[0].weight,
                                  conv_bn_relu.get_parameter('weight_conv', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[0].weight,
                                          conv_bn_relu.get_parameter('weight_conv', mode='forward'))
            assert torch.allclose(model_pytorch[0].bias,
                                  conv_bn_relu.get_parameter('bias_conv', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[0].bias,
                                          conv_bn_relu.get_parameter('bias_conv', mode='forward'))
            assert torch.allclose(model_pytorch[1].weight,
                                  conv_bn_relu.get_parameter('weight_bn', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[1].weight,
                                          conv_bn_relu.get_parameter('weight_bn', mode='forward'))
            assert torch.allclose(model_pytorch[1].bias, conv_bn_relu.get_parameter('bias_bn', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[1].bias,
                                          conv_bn_relu.get_parameter('bias_bn', mode='forward'))
            assert torch.allclose(model_pytorch[1].running_mean,
                                  conv_bn_relu.get_buffer('running_mean', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[1].running_mean, conv_bn_relu.get_buffer('running_mean'))
            assert torch.allclose(model_pytorch[1].running_var,
                                  conv_bn_relu.get_buffer('running_var', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[1].running_var, conv_bn_relu.get_buffer('running_var'))
            assert torch.allclose(model_pytorch[5].weight,
                                  linear_final.get_parameter('weight', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[5].weight,
                                          linear_final.get_parameter('weight', mode='forward'))
            assert torch.allclose(model_pytorch[5].bias, linear_final.get_parameter('bias', mode='forward')), \
                compare_pytorch_and_async(model_pytorch[5].bias,
                                          linear_final.get_parameter('bias', mode='forward'))

    @pytest.mark.parametrize('no_bn_weight_decay', [False, True])
    @pytest.mark.parametrize('weight_decay', [0.0, 5e-1])
    @pytest.mark.parametrize('momentum', [0.0, 0.9])
    @pytest.mark.parametrize('sequential', [SynchronousSequential, AsynchronousSequential])
    def test_store_vjp(self, sequential, momentum, weight_decay, no_bn_weight_decay):
        """Test the optimization of a simple model with a single layer."""
        n_in, n_hidden, n_out = 3, 7, 10

        conv_bn_relu = ConvBnReLU(n_in, n_hidden, store_vjp=False)
        linear_final = LinearFinal(n_hidden, n_out, store_vjp=False)
        model = sequential([conv_bn_relu, linear_final])
        model.synchronize_layers()

        # ----- Need to compare the updates performed by PyTorch against our custom optimizers.
        conv_bn_relu_vjp = ConvBnReLU(n_in, n_hidden, store_vjp=True)
        linear_final_vjp = LinearFinal(n_hidden, n_out, store_vjp=False)
        model_vjp = sequential([conv_bn_relu_vjp, linear_final_vjp])
        model_vjp.synchronize_layers()

        # ----- Define optimizer
        optim_kwargs = {
            'lr': 0.1,
            'momentum': momentum
        }

        if no_bn_weight_decay:
            condition = lambda name: 'bias' not in name and 'bn' not in name
            optim_kwargs['weight_decay'] = weight_decay
            model.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))
            model_vjp.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))

            condition = lambda name: 'bias' in name or 'bn' in name
            optim_kwargs['weight_decay'] = 0.
            model.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))
            model_vjp.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs, condition=condition))
        else:
            optim_kwargs['weight_decay'] = weight_decay
            model.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs))
            model_vjp.apply(lambda module: add_optimizer(module, 'sgd', optim_kwargs))

        # synchronizing model weights
        model_vjp.load_state_list(model.state_list())

        # checking with the original model
        for module, module_vjp in zip(model.modules, model_vjp.modules):
            for name, name_vjp in zip(module.list_parameters, module_vjp.list_parameters):
                assert torch.allclose(module.get_parameter(name, mode='forward'),
                                      module_vjp.get_parameter(name_vjp, mode='forward'))
                assert torch.allclose(module.get_parameter(name, mode='backward'),
                                      module_vjp.get_parameter(name_vjp, mode='backward'))
            for name, name_vjp in zip(module.list_buffers, module_vjp.list_buffers):
                assert torch.allclose(module.get_buffer(name, mode='forward'),
                                      module_vjp.get_buffer(name_vjp, mode='forward'))
                assert torch.allclose(module.get_buffer(name, mode='backward'),
                                      module_vjp.get_buffer(name_vjp, mode='backward'))

        # ----- Compare the forward and backward pass in train mode
        for idx in range(len(model.modules) * 5):
            bs = 8
            x = torch.rand(bs, n_in, 32, 32)
            y = torch.randint(0, n_out, (bs,))

            # model without store vjp
            with torch.no_grad():
                print(f'Update {idx}')
                model.forward_and_update(x, y, idx, set_grad_to_none=False)

            # model store vjp
            with torch.no_grad():
                model_vjp.forward_and_update(x, y, idx, set_grad_to_none=False)

            # checking gradients with the original model
            for k, (module, module_vjp) in enumerate(zip(model.modules, model_vjp.modules)):
                for name, name_vjp in zip(module.list_parameters, module_vjp.list_parameters):
                    preamble = f'Layer {k} - Update {idx} - Parameter {name}'

                    grad = module.get_gradient(name)
                    param_forward = module.get_parameter(name, mode='forward')
                    param_backward = module.get_parameter(name, mode='backward')

                    grad_vjp = module_vjp.get_gradient(name_vjp)
                    param_forward_vjp = module_vjp.get_parameter(name_vjp, mode='forward')
                    param_backward_vjp = module_vjp.get_parameter(name_vjp, mode='backward')

                    # check that gradients are the same between models
                    if grad is not None and grad_vjp is not None:
                        assert compare_tensors(grad, grad_vjp, preamble)

                    # check that parameters are the same between models
                    assert torch.allclose(param_forward, param_forward_vjp), \
                        compare_tensors(param_forward, param_forward_vjp, preamble)
                    assert torch.allclose(param_backward, param_backward_vjp), \
                        compare_tensors(param_backward, param_backward_vjp, preamble)

                    # compare forward and backward parameters
                    if name != 'bias_conv' and name_vjp != 'bias_conv':
                        if grad is not None and grad_vjp is not None:
                            # check that parameters backward parameters have changed
                            assert not torch.allclose(param_forward, param_backward), \
                                preamble + \
                                f' - Unchanged - Grad norm {grad.norm()} - param norm {param_forward.norm()}'
                            assert not torch.allclose(param_forward_vjp, param_backward_vjp), \
                                preamble + \
                                f' - Unchanged - Grad norm {grad_vjp.norm()} - param norm {param_forward_vjp.norm()}'
                        else:
                            # check that parameters have not changed
                            assert torch.allclose(param_forward, param_backward), \
                                compare_tensors(param_forward, param_backward, preamble)
                            assert torch.allclose(param_forward_vjp, param_backward_vjp), \
                                compare_tensors(param_forward_vjp, param_backward_vjp, preamble)

                for name, name_vjp in zip(module.list_buffers, module_vjp.list_buffers):
                    preamble = f'Layer {k} - Update {idx} - Buffer {name}'

                    buffer_forward = module.get_buffer(name, mode='forward')
                    buffer_backward = module.get_buffer(name, mode='backward')

                    buffer_forward_vjp = module_vjp.get_buffer(name_vjp, mode='forward')
                    buffer_backward_vjp = module_vjp.get_buffer(name_vjp, mode='backward')

                    # check that buffers are the same between models
                    assert torch.allclose(buffer_forward, buffer_forward_vjp), \
                        compare_tensors(buffer_forward, buffer_forward_vjp, preamble)
                    assert torch.allclose(buffer_backward, buffer_backward_vjp), \
                        compare_tensors(buffer_backward, buffer_backward_vjp, preamble)

            # reset gradients
            for module, module_vjp in zip(model.modules, model_vjp.modules):
                module.set_grad_to_none()
                module_vjp.set_grad_to_none()

            # synchronize forward and backward parameters
            model.synchronize_layers()
            model_vjp.synchronize_layers()

            for module, module_vjp in zip(model.modules, model_vjp.modules):
                for name, name_vjp in zip(module.list_buffers, module_vjp.list_buffers):
                    preamble = f'Layer {k} - Update {idx} - Buffer {name}'

                    buffer = module.get_buffer(name, mode='forward')
                    buffer_vjp = module_vjp.get_buffer(name_vjp, mode='forward')

                    # check that buffers are the same between models
                    assert torch.allclose(buffer, buffer_vjp), \
                        compare_tensors(buffer, buffer_vjp, preamble)
