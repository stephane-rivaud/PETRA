import torch
from async_torch.layers.layers import AsynchronousGenericLayer


class TestParameterManagement:
    def test_parameters(self):
        model = AsynchronousGenericLayer()

        # register parameter
        name, value = 'weight', torch.randn(5)
        model._register_parameters(name, value)
        assert name in model.list_parameters
        assert hasattr(model, name + '_forward')
        assert hasattr(model, name + '_backward')
        assert hasattr(model, name + '_grad_backward')

        assert torch.allclose(model.get_parameter(name, mode='forward'), value)
        assert torch.allclose(model.get_parameter(name, mode='backward'), value)
        assert model.get_gradient(name) is None

        # delete parameter
        model._delete_parameters(name)
        assert name not in model.list_parameters
        assert not hasattr(model, name + '_forward')
        assert not hasattr(model, name + '_backward')
        assert not hasattr(model, name + '_grad_backward')

        # register buffer
        name, value = 'running_mean', torch.zeros(5)
        model._register_buffers(name, value)
        assert name in model.list_buffers
        assert hasattr(model, name + '_forward')
        assert hasattr(model, name + '_backward')

        # delete parameter
        model._delete_buffers(name)
        assert name not in model.list_buffers
        assert not hasattr(model, name + '_forward')
        assert not hasattr(model, name + '_backward')
