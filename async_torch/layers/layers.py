import torch

from .buffer_memory import CachingBufferSimple
from .compression import QuantizSimple


class AsynchronousGenericLayer:
    def __init__(self, first_layer=False, quantizer=QuantizSimple, store_input=True, store_param=True, store_vjp=False,
                 accumulation_steps=1, accumulation_averaging=True, approximate_input=False):

        if approximate_input:
            print('Approximate input is True, store_input is set to False, store_param is set to True and store_vjp is set to False')
            store_input = False
            store_param = True
            store_vjp = False

        if store_vjp:
            print('Store vjp is True, store_input is set to False, store_param is set to False and approximate_input is set to False')
            store_input = False
            store_param = False
            approximate_input = False

        self.list_parameters = []
        self.list_buffers = []
        self.buffer = CachingBufferSimple()
        self.first_layer = first_layer
        self.quantizer = quantizer()
        self.store_input = store_input
        self.store_param = store_param
        self.store_vjp = store_vjp
        self.accumulation_steps = accumulation_steps
        self.accumulation_averaging = accumulation_averaging
        self.approximate_input = approximate_input

        self.count = 0
        self.synchronize_buffers = False
        self.optimizers = []

    def state_list(self):
        param_forward = {attr: self.get_parameter(attr, mode='forward').clone() for attr in self.list_parameters}
        param_backward = {attr: self.get_parameter(attr, mode='backward').clone() for attr in self.list_parameters}
        buffer_forward = {attr: self.get_buffer(attr, mode='forward').clone() for attr in self.list_buffers}
        buffer_backward = {attr: self.get_buffer(attr, mode='backward').clone() for attr in self.list_buffers}
        optimizers_attributes = [optimizer.state_list() for optimizer in self.optimizers]
        return param_forward, param_backward, buffer_forward, buffer_backward, optimizers_attributes

    def load_state_list(self, state):
        param_forward, param_backward, buffer_forward, buffer_backward, optimizers_attributes = state
        for attr, value in param_forward.items():
            self.get_parameter(attr, mode='forward').data.copy_(value)
        for attr, value in param_backward.items():
            self.get_parameter(attr, mode='backward').data.copy_(value)
        for attr, value in buffer_forward.items():
            self.get_buffer(attr, mode='forward').data.copy_(value)
        for attr, value in buffer_backward.items():
            self.get_buffer(attr, mode='backward').data.copy_(value)
        for optimizer, optim_state in zip(self.optimizers, optimizers_attributes):
            optimizer.load_state_list(optim_state)

    def _register_parameters(self, name, value):
        # Prevent from registering the same parameter twice
        if name not in self.list_parameters:
            self.list_parameters.append(name)

        # We create the forward value and copy it
        setattr(self, name + '_forward', value.clone())

        # We create the backward value and copy it
        setattr(self, name + '_backward', value.clone())

        # We create the gradient tensor
        setattr(self, name + '_grad_backward', None)

    def set_parameter(self, name, value, mode='forward'):
        assert name in self.list_parameters, f'No parameter named {name}'
        assert mode in ['forward', 'backward']
        attr_name = name + '_' + mode
        setattr(self, attr_name, value)

    def _delete_parameters(self, name):
        # We need to remove the name from the parameter list
        if name not in self.list_parameters:
            raise ValueError('no parameter named', name)
        else:
            self.list_parameters.remove(name)

        # We delete the forward value
        delattr(self, name + '_forward')

        # We delete the backward value
        delattr(self, name + '_backward')

        # We delete the gradient tensor
        delattr(self, name + '_grad_backward')

    def _register_buffers(self, name, value):
        # Prevent from registering the same buffer twice
        if name not in self.list_buffers:
            self.list_buffers.append(name)

        # We create the forward value and copy it
        setattr(self, name + '_forward', value.clone())

        # We create the backward value and copy it
        setattr(self, name + '_backward', value.clone())

    def set_buffer(self, name, value, mode='forward'):
        assert name in self.list_buffers, f'No buffer named {name}'
        assert mode in ['forward', 'backward']
        attr_name = name + '_' + mode
        setattr(self, attr_name, value)

    def _delete_buffers(self, name):
        # We need to remove the name from the buffer list
        if name not in self.list_buffers:
            raise ValueError('no buffer named', name)
        else:
            self.list_buffers.remove(name)

        # We delete the forward value
        delattr(self, name + '_forward')

        # We delete the backward value
        delattr(self, name + '_backward')

    def get_parameter(self, name, mode='forward'):
        assert name in self.list_parameters, f'No parameter named {name}'
        assert mode in ['forward', 'backward']
        attr_name = name + '_' + mode
        return getattr(self, attr_name)

    def get_buffer(self, name, mode='forward'):
        assert name in self.list_buffers, f'No buffer named {name}'
        assert mode in ['forward', 'backward']
        attr_name = name + '_' + mode
        return getattr(self, attr_name)

    def get_gradient(self, name):
        assert name in self.list_parameters, f'No parameter named {name}'
        attr_name = name + '_grad_backward'
        return getattr(self, attr_name)

    def set_gradient(self, name, value):
        assert name in self.list_parameters, f'No parameter named {name}'
        attr_name = name + '_grad_backward'
        setattr(self, attr_name, value)

    def set_grad_to_none(self, name=None):
        if name is None:
            for name in self.list_parameters:
                self.set_gradient(name, None)
        else:
            self.set_gradient(name, None)

    def local_f(self, input, *parameters_buffers):
        """ This function should be a wrapper which processes parameters
        and buffers, as they appear in self.list_parameters and self.list_buffers.
        It should return the output of the forward
        """
        raise NotImplementedError('This function should be implemented in the subclass')

    def local_f_reversed(self, output, *parameters_buffers):
        """ This function should be a wrapper which processes parameters
        and buffers, as they appear in self.list_parameters and self.list_buffers.
        It should reconstruct the input of the forward from the output of the forward.
        """
        raise NotImplementedError('This function should be implemented in the subclass')

    def set_lr(self, learning_rate):
        for optimizer in self.optimizers:
            optimizer.set_lr(learning_rate)

    def to(self, *args, **kwargs):
        self.to_forward(*args, **kwargs)
        self.to_backward(*args, **kwargs)

    def to_forward(self, *args, **kwargs):
        for name in self.list_parameters:
            self.set_parameter(name, self.get_parameter(name, mode='forward').to(*args, **kwargs), mode='forward')

        for name in self.list_buffers:
            self.set_buffer(name, self.get_buffer(name, mode='forward').to(*args, **kwargs), mode='forward')

    def to_backward(self, *args, **kwargs):
        for name in self.list_parameters:
            self.set_parameter(name, self.get_parameter(name, mode='backward').to(*args, **kwargs), mode='backward')

        for name in self.list_buffers:
            self.set_buffer(name, self.get_buffer(name, mode='backward').to(*args, **kwargs), mode='backward')

        for optimizer in self.optimizers:
            optimizer.to(*args, **kwargs)

    def forward(self, input, input_id=None):
        """
        Forward function which process inputs. This function is core to the async-backpropagation
        algorithm as it will dequantize some inputs, potentially store them for backward and then
        process forward using local parameters.

        Parameters
        ----------
        input : tuple of torch.tensor or a torch.tensor
            a list which corresponds to the inputs to process
        input_id : int (default: None)
            in backward mode, allows to store the id of a given batch of data

        Returns
        -------
        output : tuple of torch.tensor or a torch.tensor
            the output of the corresponding forward pass
        """
        training = input_id is not None  # in this case, this is train mode
        # print(f'Forward function -- training: {training} -- input_id: {input_id}')

        if training and not self.store_vjp:
            store = tuple()
            if self.store_input:
                # print('store input')
                store += (input,)
            if self.store_param:
                # print('store param')
                store += tuple([self.get_parameter(name, mode='forward').clone() for name in self.list_parameters])
            store += (input_id,)
            self.buffer.add(self.quantizer.quantize_buffer_forward(store))

        if not self.first_layer:
            input = self.quantizer.dequantize_input_forward(input)

        # define the local function
        list_parameter_forward = [self.get_parameter(name, mode='forward').clone() for name in self.list_parameters]
        list_buffer_forward = [self.get_buffer(name, mode='forward') for name in self.list_buffers]
        local_f = lambda *input_param: self.local_f(*input_param, *list_buffer_forward, training)

        # forward pass
        if training and self.store_vjp:
            # print('store-vjp option is active')
            # process the forward pass while creating vjp
            if isinstance(input, tuple):
                n_input = len(input)
                output, vjpfunc = torch.func.vjp(local_f, *input, *list_parameter_forward)
            elif torch.is_tensor(input):
                n_input = 1
                output, vjpfunc = torch.func.vjp(local_f, input, *list_parameter_forward)
            else:
                raise TypeError('Input should be a tensor or a tuple of tensors.')
        else:
            # print('store-vjp option is not active')
            # process the forward pass without vjp
            if isinstance(input, tuple):
                output = local_f(*input, *list_parameter_forward)
            elif torch.is_tensor(input):
                output = local_f(input, *list_parameter_forward)
            else:
                raise TypeError('Input should be a tensor or a tuple of tensors.')

        # store vjp in the buffer
        if training and self.store_vjp:
            # print('storing store-vjp')
            store = (vjpfunc,)
            store += tuple([buffer.clone() for buffer in list_buffer_forward])
            store += (n_input, input_id)
            self.buffer.add(store)

        # quantize output
        output = self.quantizer.quantize_output_forward(output)

        # print()
        return output

    def backward(self, output, grad_output, input_id):
        """
        Backward function which process gradients (grad_output and grad_input).
        This function is core to the async-backpropagation algorithm as it will
        get the context (i.e., activations and parameters) related to an initial
        forward pass. It will potentially dequantize activations and compute
        gradients using a vjp function, meaning that local activations are
        recomputed.

        Parameters
        ----------
        output
        grad_output : tuple of torch.tensor or a torch.tensor
            a list which corresponds to the gradient_outputs which will be processed
        input_id : int
            allows to recover a batch given the id

        Returns
        -------
        grad_input : tuple of torch.tensor or a torch.tensor
            the output of the corresponding backward pass
        input_id : int
            allows to recover a batch given the id, which is propagated to the next layer
        """
        # print('Backward function -- input_id: ', input_id)
        grad_output = self.quantizer.dequantize_grad_output_backward(grad_output)

        buffer = self.buffer.get()

        # input_id
        input_id2 = buffer[-1]
        assert (input_id == input_id2)

        # vjp function
        if self.store_vjp:
            # print('store-vjp option is active')
            vjpfunc = buffer[0]

            start_index = 1
            end_index = start_index + len(self.list_buffers)
            list_buffer_forward = list(buffer[start_index:end_index])

            n_input = buffer[-2]
            input = None

            assert end_index == len(buffer) - 2, f'Buffer size mismatch: {end_index} vs {len(buffer) - 2}'

            # update buffers
            for name, buffer_forward in zip(self.list_buffers, list_buffer_forward):
                self.get_buffer(name, mode='backward').data.copy_(buffer_forward)

        else:
            # parameters
            if self.store_param:
                # print('store param: retrieving forward weights from buffer to compute jacobian')
                start_index = 1 if self.store_input else 0
                end_index = start_index + len(self.list_parameters)
                list_parameters = list(buffer[start_index:end_index])
                assert end_index == len(buffer) - 1, f'Buffer size mismatch: {end_index} vs {len(buffer) - 1}'
            else:
                # print('no store param: using backward weights to compute jacobian')
                list_parameters = [self.get_parameter(name, mode='backward') for name in self.list_parameters]

            # buffers
            list_buffer_backward = [self.get_buffer(name, mode='backward') for name in self.list_buffers]

            # input
            if self.store_input:
                # print('store input: retrieving forward input from buffer')
                input = buffer[0]
            else:
                if self.approximate_input:
                    # print('approximate input: reconstructing input from output with backward parameters')
                    list_param_reverse = [self.get_parameter(name, mode='backward') for name in self.list_parameters]
                else:
                    # print('no store input: reconstructing input from output according to store param option')
                    list_param_reverse = list_parameters
                if isinstance(output, tuple):
                    input = self.local_f_reversed(*output, *list_param_reverse, *list_buffer_backward, True)
                elif torch.is_tensor(output):
                    input = self.local_f_reversed(output, *list_param_reverse, *list_buffer_backward, True)
                else:
                    raise TypeError('Output should be a tensor or a tuple of tensors.')
            n_input = len(input) if isinstance(input, tuple) else 1

            # define the local function
            local_f = lambda *input_param: self.local_f(*input_param, *list_buffer_backward, True)

            # compute vjp
            if torch.is_tensor(input):
                output, vjpfunc = torch.func.vjp(local_f, input, *list_parameters)
            elif isinstance(input, tuple):
                output, vjpfunc = torch.func.vjp(local_f, *input, *list_parameters)
            else:
                raise TypeError('Input should be a tensor or a tuple of tensors.')
            del output

        # compute gradients
        grads = vjpfunc(grad_output)

        if n_input > 1:
            grad_input = tuple(grads[0:n_input])
        else:
            grad_input = grads[0]
        grad_parameters = grads[n_input:]

        # update gradients
        for i, name in enumerate(self.list_parameters):
            grad = self.get_gradient(name)
            if grad is not None:
                self.set_gradient(name, grad + grad_parameters[i])
            else:
                self.set_gradient(name, grad_parameters[i])

        grad_input = self.quantizer.quantize_grad_input_backward(grad_input)

        self.last_id = input_id
        return input, grad_input, input_id

    def synchronize_forwardbackward(self):
        """
        This method copies the backward parameters to the forward parameters
        and the buffers.
        """

        for name in self.list_parameters:
            data_backward = self.get_parameter(name, mode='backward').data
            self.get_parameter(name, mode='forward').data.copy_(data_backward)

        if self.synchronize_buffers:
            for name in self.list_buffers:
                data_backward = self.get_buffer(name, mode='backward').data
                self.get_buffer(name, mode='forward').data.copy_(data_backward)

    def forward_with_decorations(self, input, label, input_id):
        output = self.forward(input, input_id)
        return output, label, input_id

    def update(self, set_grad_to_none=True):
        self.count = (self.count + 1) % self.accumulation_steps
        if self.count == 0:
            for optimizer in self.optimizers:
                list_param_backward = []
                list_grad_backward = []
                for k, name in enumerate(optimizer.list_parameters):
                    attr_value = self.get_parameter(name, mode='backward')
                    attr_grad = self.get_gradient(name)
                    if attr_grad is not None:
                        if self.accumulation_averaging:
                            attr_grad /= self.accumulation_steps
                        list_param_backward.append(attr_value)
                        list_grad_backward.append(attr_grad)
                optimizer.update(list_param_backward, list_grad_backward)

                if set_grad_to_none:
                    # Since we updated, we remove accumulation.
                    for name in optimizer.list_parameters:
                        self.set_grad_to_none(name)


class AsynchronousFinal(AsynchronousGenericLayer):
    def __init__(self, *args, **kwargs):
        super(AsynchronousFinal, self).__init__(*args, **kwargs)

    def loss(self, x, y):
        """
        This method computes the loss between a prediction x and a target y.
        It should be implemented by all subclasses.
        """
        pass

    def forward_and_backward(self, input, label, input_id=None):
        if not self.first_layer:
            input = self.quantizer.dequantize_input_forward(input)

        self.synchronize_forwardbackward()  # since it is the last layer, parameters forward=parameters backward

        # note: the running mean won't be used at test time and are backward's ones.
        list_parameter_backward = [self.get_parameter(name, mode='backward') for name in self.list_parameters]
        list_buffer_backward = [self.get_buffer(name, mode='backward') for name in self.list_buffers]
        training = input_id is not None  # in this case, this is train mode
        local_f = lambda *input_param: self.local_f(*input_param, *list_buffer_backward, training)

        # forward pass
        if torch.is_tensor(input):
            (pred, vjpfunc) = torch.func.vjp(local_f, input, *list_parameter_backward)
        elif isinstance(input, tuple):
            (pred, vjpfunc) = torch.func.vjp(local_f, *input, *list_parameter_backward)
        else:
            raise TypeError('Input should be a tensor or a tuple of tensors.')

        # compute loss
        loss_fn = lambda x: self.loss(x, label)
        (loss, vjploss) = torch.func.vjp(loss_fn, pred)

        # compute gradients
        grad_loss = vjploss(torch.ones_like(loss))[0]
        grads = vjpfunc(grad_loss)
        del grad_loss

        # assign input gradients
        if isinstance(input, tuple):
            n_input = len(input)
            grad_input = tuple(grads[:n_input])
        else:
            n_input = 1
            grad_input = grads[0]

        # assign parameter gradients
        grad_parameters = grads[n_input:]

        # update gradients
        for i, name in enumerate(self.list_parameters):
            grad = self.get_gradient(name)
            if grad is not None:
                self.set_gradient(name, grad + grad_parameters[i])
            else:
                self.set_gradient(name, grad_parameters[i])

        grad_input = self.quantizer.quantize_grad_input_backward(grad_input)
        return input, grad_input, [loss, input_id, pred, label]
