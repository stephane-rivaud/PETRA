import torch
import sys


def get_quantizer(quant_name):
    current_module = sys.modules[__name__]
    if hasattr(current_module, quant_name):
        return getattr(current_module, quant_name)
    else:
        raise ValueError(f"No class named {quant_name} in this module.")


class QuantizGeneric:
    def __init__(self, tol_activation, tol_gradient, tol_buffer):
        self.tol_activation = tol_activation
        self.tol_gradient = tol_gradient
        self.tol_buffer = tol_buffer

    def quantize_backward_communication(self, x):
        """
        This function is used in the backward pass to compress gradients
        which comes from the past layer. Elements which are not tensors
        or tuples of tensors are not quantized.

        Parameters
        ----------
        x : tuple of (torch.tensor, tuples of torch.tensor or other quantities)
            the gradient tensors and potentially other informations passed by the backward.
            
        Returns
        -------
        x_compressed : tuple of the same size as x
            any compressed formats to encode the pytorch tensor of x.
        """
        pass

    def dequantize_backward_communication(self, x):
        """
        This function is used in the backward pass to decompress tuples of gradients
        which will be sent to the next layer. 

        Parameters
        ----------
        x : tuple of compressed tensors
            a tuple of elements to decompress.
            
        Returns
        -------
        x : tuple of the same size as x_compressed
            transforms back into decompressed elements of th same size as x_compressed.
        """
    
    def dequantize_forward_communication(self, x):
        """
        This function is used in the forward pass to decompress tuples of compressed
        activations which comes from the previous layer.

        Parameters
        ----------
        x : tuple of compressed tensors
            a tuple of elements to decompress.
            
        Returns
        -------
        x : tuple of the same size as x_compressed
            transforms back into decompressed elements of th same size as x_compressed.
        """
    
    def quantize_forward_communication(self, x):
        """
        This function is used in the forward pass to compress activations
        which comes from the past layer. Elements which are not tensors
        or tuples of tensors are not quantized.

        Parameters
        ----------
        x : tuple of (torch.tensor, tuples of torch.tensor or other quantities)
            the activations and potentially other informations passed by the forward.
            
        Returns
        -------
        x_compressed : tuple of the same size as x
            any compressed formats to encode the pytorch tensor of x.
        """

    def dequantize_buffer_backward(self, buffer):
        """
        This function is used in the backward pass to decompress tuples stored in buffers
        which will be sent to the next layer. 

        Parameters
        ----------
        buffer : tuple of compressed tensors
            a tuple of elements to decompress.
            
        Returns
        -------
        buffer : tuple of the same size as x_compressed
            transforms back into decompressed elements of th same size as x_compressed.
        """
        pass

    def quantize_buffer_forward(self, buffer):
        """
        This function is used in the forward pass to compress buffers.
        Elements which are not tensors or tuples of tensors are not quantized.

        Parameters
        ----------
        buffer : tuple of (torch.tensor, tuples of torch.tensor or other quantities)
            the buffer elements and potentially other informations passed by the forward.
            
        Returns
        -------
        x_compressed : tuple of the same size as x
            any compressed formats to encode the pytorch tensor of x.
        """
        pass


class QuantizAll(QuantizGeneric):
    def __init__(self, tol):
        tol_activation = tol
        tol_gradient = tol
        tol_buffer = tol
        super(QuantizAll, self).__init__(tol_activation, tol_gradient, tol_buffer)

    def _quantize(self, x):
        pass

    def _dequantize(self, x):
        pass

    def quantize_backward_communication(self, x):
        if torch.is_tensor(x):
            return self._quantize(x)
        elif isinstance(x, tuple):
            return tuple(self._quantize(xx) for xx in x)

    def dequantize_backward_communication(self, x):
        if torch.is_tensor(x):
            return self._dequantize(x)
        elif isinstance(x, tuple):
            return tuple(self._dequantize(xx) for xx in x)
    
    def dequantize_forward_communication(self, x):
        if torch.is_tensor(x):
            return self._dequantize(x)
        elif isinstance(x, tuple):
            return tuple(self._dequantize(xx) for xx in x)
    
    def quantize_forward_communication(self, x):
        if torch.is_tensor(x):
            return self._quantize(x)
        elif isinstance(x, tuple):
            return tuple(self._quantize(xx) for xx in x)
        
    def dequantize_buffer_backward(self, buffer):
        if torch.is_tensor(buffer):
            return self._dequantize(buffer)
        elif isinstance(buffer, tuple):
            return tuple(self._dequantize(x) for x in buffer)

    def quantize_buffer_forward(self, buffer):
        if torch.is_tensor(buffer):
            return self._quantize(buffer)
        elif isinstance(buffer, tuple):
            return tuple(self._quantize(x) for x in buffer)


class Quantiz16Bits(QuantizAll):
    def __init__(self):
        tol = 2**-11 # 5 bit exposant, 10+1 bits sign+ fraction 
        super(Quantiz16Bits, self).__init__(tol)

    def _quantize(self, x):
        if torch.is_tensor(x):
            return x.to(torch.float16)
        elif isinstance(x, tuple):
            return tuple(self._quantize(xx) for xx in x)
        else:
            return x

    def _dequantize(self, x):
        if torch.is_tensor(x):
            return x.to(torch.float32)
        elif isinstance(x, tuple):
            return tuple(self._dequantize(xx) for xx in x)
        else:
            return x


class QuantizSimple(QuantizAll):
    def __init__(self):
        tol = 0
        super(QuantizSimple, self).__init__(tol)

    def _quantize(self, x):
        return x

    def _dequantize(self, x):
        return x


class Quantiz8Bits(QuantizAll): # compression
    def __init__(self, *args, **kwargs):
        self.range = 2 ** 8 - 1
        tol = 2**-7
        super(Quantiz8Bits, self).__init__(tol, *args, **kwargs)

    def _quantize(self, x):
        if torch.is_tensor(x):
            x_sgn = torch.sign(x)
            x_abs = x.abs()
            x_norm = x_abs.max()
            x_scaled = x_abs / x_norm
            x_prequant = x_scaled * self.range + 0.5
            x_quant = x_prequant.to(torch.uint8)
            return {
                'x_quant': x_quant,
                'x_sgn': x_sgn,
                'x_norm': x_norm
            }
        elif isinstance(x, tuple):
            return tuple(self._quantize(xx) for xx in x)
        else:
            return x

    def _dequantize(self, x):
        if isinstance(x, dict):
            x_quant = x['x_quant']
            x_sgn = x['x_sgn']
            x_norm = x['x_norm']
            x_prequant = x_quant.to(torch.float32)
            x_scaled = x_prequant / self.range
            x_abs = x_scaled * x_norm
            xhat = x_abs * x_sgn
            return xhat
        elif isinstance(x, tuple):
            return tuple(self._dequantize(xx) for xx in x)
        else:
            return x


class QuantizQSGD(QuantizAll):
    def __init__(self, *args, **kwargs):
        tol = 2**-2
        super(QuantizQSGD, self).__init__(tol, *args, **kwargs)

    def _quantize(self, x):
        if torch.is_tensor(x):
            n = 8
            x = x.float()
            x_norm = torch.norm(x, p=float('inf'))

            sgn_x = ((x > 0).float() - 0.5) * 2

            p = torch.div(torch.abs(x), x_norm)
            renormalize_p = torch.mul(p, n)
            floor_p = torch.floor(renormalize_p)
            compare = torch.rand_like(floor_p)
            final_p = renormalize_p - floor_p
            margin = (compare < final_p).float()
            xi = (floor_p + margin) / n

            x_tilde = x_norm * sgn_x * xi
            return x_tilde
        else:
            return x

    def _dequantize(self, x):
        return x
