import pytest
import torch

from async_torch.layers.layers import AsynchronousGenericLayer
from async_torch.layers.compression import get_quantizer, QuantizSimple


class AddConstant(AsynchronousGenericLayer):
    def __init__(self, constant, *args, **kwargs):
        super(AddConstant, self).__init__(*args, **kwargs)
        self._register_parameters('bias', constant.clone())

    def local_f(self, x, constant, training):
        return x + constant


class SumAddConstant(AsynchronousGenericLayer):
    def __init__(self, constant, *args, **kwargs):
        super(SumAddConstant, self).__init__(*args, **kwargs)
        self._register_parameters('constant', constant.clone())

    def local_f(self, x, tilde_x, constant, training):
        return x + tilde_x + constant


class TestCompressionEffects:
    @pytest.mark.parametrize("quantizer_name", ['QuantizSimple', 'Quantiz16Bits' 'Quantiz8Bits'])
    @pytest.mark.parametrize(
        'stuff',
        [
            (AddConstant, torch.randn(3, 2, 4), torch.randn(3, 2, 4), torch.randn(3, 2, 4)),
            (SumAddConstant, (torch.randn(3, 2, 4), torch.randn(3, 2, 4)), torch.randn(3, 2, 4), torch.randn(3, 2, 4))
            ]
        )
    def test_fake_module(self, stuff, quantizer_name):
        module, x, constant, grad_output = stuff
        quantizer = get_quantizer(quantizer_name)
        mod = module(constant, quantizer=QuantizSimple, first_layer=True)
        mod_comp = module(constant, quantizer=quantizer, first_layer=True)

        # ----- compare forward pass between compressed and not compressed input -----
        # eval mode
        y = mod.forward(x)
        y_comp = mod_comp.forward(x)

        y_hat = mod_comp.quantizer.dequantize_forward_communication(y_comp)
        # note: you need to use both atol and rtol due to instabilities around 0 (it's classical)
        assert torch.allclose(y_hat, y, atol=mod_comp.quantizer.tol_activation, rtol=mod_comp.quantizer.tol_activation)

        # train mode
        idx = 0
        y = mod.forward(x, idx)
        y_comp = mod_comp.forward(x, idx)

        y_hat = mod_comp.quantizer.dequantize_forward_communication(y_comp)
        assert torch.allclose(y_hat, y, atol=mod_comp.quantizer.tol_activation, rtol=mod_comp.quantizer.tol_activation)

        # ----- compare compressed and not compressed buffer -----
        b = mod.buffer.get()
        b_comp = mod_comp.buffer.get()
        b_hat = mod_comp.quantizer.dequantize_buffer_backward(b_comp)

        if torch.is_tensor(b):
            assert torch.is_tensor(b) and torch.is_tensor(b_hat), \
                "b and b_hat should be tensors ({}, {})".format(len(b), len(b_hat))
            assert torch.allclose(b_hat, b, atol=mod_comp.quantizer.tol_buffer, rtol=mod_comp.quantizer.tol_buffer)
        elif isinstance(b, tuple):
            assert len(b) == len(b_hat), "b and b_hat should have the same length ({}, {})".format(len(b), len(b_hat))
            for f, f_hat in zip(b, b_hat):
                if torch.is_tensor(f):
                    assert torch.is_tensor(f_hat)
                    assert torch.allclose(f_hat, f, atol=mod_comp.quantizer.tol_buffer,
                                          rtol=mod_comp.quantizer.tol_buffer)

        # ----- compare compressed and not compressed gradients -----
        # We put back the element in the buffer...
        mod.buffer.add(b)
        mod_comp.buffer.add(b)  # We put the same element not compressed... to check only the gradients compression

        _, grad_b, _ = mod.backward(None, grad_output, 0)
        _, grad_comp_b, _ = mod_comp.backward(None, grad_output, 0)
        grad_hat_b = mod_comp.quantizer.dequantize_backward_communication(grad_comp_b)
        if isinstance(grad_b, tuple):
            for f, f_hat in zip(grad_b, grad_hat_b):
                assert torch.allclose(f_hat, f, atol=mod_comp.quantizer.tol_buffer, rtol=mod_comp.quantizer.tol_buffer)
        else:
            assert torch.allclose(grad_b, grad_hat_b, atol=mod_comp.quantizer.tol_gradient,
                                  rtol=mod_comp.quantizer.tol_gradient)
