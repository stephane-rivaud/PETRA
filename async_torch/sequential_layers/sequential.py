import numpy as np
import torch


# Utility for pushing lists to device
# TODO: Possibly move to utils file
def _to_device(item, device):
    # This function will need to be customized based on the structure of the data items
    if isinstance(item, (list, tuple)):
        return [_to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: _to_device(v, device) for k, v in item.items()}
    elif torch.is_tensor(item):
        return item.to(device, non_blocking=True)
    else:
        return item


class Sequential:
    def __init__(self, modules):
        self.modules = modules

    def state_list(self):
        state_list = []
        depth = len(self.modules)
        for i in range(depth):
            state_list.append(self.modules[i].state_list())
        return state_list

    def load_state_list(self, state):
        depth = len(self.modules)
        for i in range(depth):
            self.modules[i].load_state_list(state[i])

    def synchronize_layers(self):
        depth = len(self.modules)
        for i in range(depth):
            self.modules[i].synchronize_forwardbackward()

    def to(self, *args, **kwargs):
        for module in self.modules:
            module.to(*args, **kwargs)
        return self

    def set_lr(self, learning_rate):
        for x in self.modules:
            x.set_lr(learning_rate)

    def apply(self, f):
        for x in self.modules:
            f(x)

    def forward(self, x, idx=None, targets=None):
        depth = len(self.modules)
        modules = self.modules
        for i in range(depth):
            x = modules[i].forward(x, idx)
        loss = modules[depth - 1].loss(x, targets) if targets is not None else None
        return loss, x

    def update(self, set_grad_to_none=True):
        for module in self.modules:
            module.update(set_grad_to_none=set_grad_to_none)

    def set_grad_to_none(self):
        for module in self.modules:
            module.set_grad_to_none()


class SynchronousSequential(Sequential):
    def __init__(self, modules):
        super(SynchronousSequential, self).__init__(modules)

    def forward_and_update(self, x, y, idx, set_grad_to_none=True, update=True):
        depth = len(self.modules)
        modules = self.modules
        for i in range(depth - 1):
            x = modules[i].forward(x, idx)

        input_from_backward, grad_x, meta = modules[depth - 1].forward_and_backward(x, y, idx)
        L, idx, x, _ = meta
        for i in range(depth - 2, -1, -1):
            input_from_backward, grad_x, _ = modules[i].backward(input_from_backward, grad_x, idx)

        if update:
            self.update(set_grad_to_none=set_grad_to_none)
        return L, x, y


class AsynchronousSequential(Sequential):
    def __init__(self, modules):
        super(AsynchronousSequential, self).__init__(modules)
        self.output = {}
        self.input_from_backward = {}
        self.output_idx = {}
        self.grad_input = {}
        self.label_output = {}
        self.grad_input_idx = {}

    def forward_and_update(self, x, y, idx, set_grad_to_none=True, update=True):
        depth = len(self.modules)
        modules = self.modules

        input = {}
        input_idx = {}
        label_input = {}

        output = self.output
        output_idx = self.output_idx
        label_output = self.label_output

        input_from_backward = self.input_from_backward
        grad_input = self.grad_input
        grad_input_idx = self.grad_input_idx

        output_for_backward = {}
        grad_output = {}
        grad_output_idx = {}

        # Prepare the input for each "asynchronous" block
        for i in range(depth):
            # if i == 0 and idx != -1:
            if i == 0:
                input[0], input_idx[0], label_input[0] = x.clone(), idx, y.clone()
            elif (i - 1) in output:
                input[i], input_idx[i], label_input[i] = output[i - 1], output_idx[i - 1], label_output[i - 1]
                del output[i - 1]
                del output_idx[i - 1]
                del label_output[i - 1]

        # Prepare the grad_output for each "asynchronous" block
        for i in range(depth - 1):
            if (i + 1) in grad_input and (i + 1) in input_from_backward:
                output_for_backward[i], grad_output[i], grad_output_idx[i] \
                    = input_from_backward[i + 1], grad_input[i + 1], grad_input_idx[i + 1]
                del input_from_backward[i + 1]
                del grad_input[i + 1]
                del grad_input_idx[i + 1]

        # Apply each "asynchronous" block on the input, forward mode
        L = 0

        for i in range(depth):
            if i in input:
                if i < depth - 1:
                    output[i], label_output[i], output_idx[i] \
                        = modules[i].forward_with_decorations(input[i], label_input[i], input_idx[i])
                else:
                    input_from_backward[depth - 1], grad_input[depth - 1], meta \
                        = modules[i].forward_and_backward(input[i], label_input[i], input_idx[i])
                    L, grad_input_idx[depth - 1], output[depth - 1], label_output[depth - 1] = meta

        # Apply each "asynchronous" block on the grad_output, backward mode -- except the last layer
        for i in range(depth - 1):
            if i in grad_output:
                input_from_backward[i], grad_input[i], grad_input_idx[i] \
                    = modules[i].backward(output_for_backward[i], grad_output[i], grad_output_idx[i])
                if i == 0:
                    del input_from_backward[i], grad_input[i], grad_input_idx[i]

        # update
        if update:
            self.update(set_grad_to_none=set_grad_to_none)

        # If no output has been computed, handle it with a None.        
        if depth - 1 in output:
            return L, output[depth - 1], label_output[depth - 1]
        else:
            return L, None, None


class AsynchronousParallel(Sequential):
    def __init__(self, modules, n_devices=1):
        super(AsynchronousParallel, self).__init__(modules)
        import warnings
        warnings.warn("The module AsynchronousParallel is still experimental", RuntimeWarning, stacklevel=2)
        self.output = {}
        self.output_idx = {}
        self.grad_input = {}
        self.label_output = {}
        self.grad_input_idx = {}
        self.depth = len(self.modules)
        gpus = np.round(np.linspace(0, n_devices - 1, len(self.modules))).astype(int)

        for i in range(len(self.modules)):
            self.modules[i].to('cuda:' + str(gpus[i]))
        gpus_ = gpus[:self.depth - 1]
        self.gpus = np.concatenate((gpus, gpus_[::-1]))

        print('GPUs organization:' + str(self.gpus))

        self.streams = [torch.cuda.Stream(device=self.gpus[i]) for i in range(self.depth)] + [
            torch.cuda.Stream(device=self.gpus[i]) for i in range(self.depth - 2, -1, -1)]

    def forward(self, x):
        depth = len(self.modules)
        modules = self.modules
        for i in range(depth):
            device = 'cuda:' + str(self.gpus[i])
            x = modules[i].forward(_to_device(x, device))
        return x.to('cuda:' + str(self.gpus[0]))

    def forward_and_update(self, x, y, idx):
        depth = len(self.modules)
        modules = self.modules
        input = {}
        output = self.output
        input_idx = {}
        output_idx = self.output_idx
        grad_input = self.grad_input
        grad_output = {}
        label_input = {}
        label_output = self.label_output
        grad_input_idx = self.grad_input_idx
        grad_output_idx = {}

        for i in range(depth):
            device = 'cuda:' + str(self.gpus[i])
            if i == 0 and idx != -1:
                input[0], input_idx[0], label_input[0] = x.to(device), idx, y.to(device)
            if (i - 1) in output:
                if isinstance(output[i - 1], list):
                    input[i] = [self._send_to(tensor, device) for tensor in output[i - 1]]
                else:
                    input[i] = self._send_to(output[i - 1], device)
                input_idx[i], label_input[i] = output_idx[i - 1], label_output[i - 1].to(device)
                del output[i - 1], output_idx[i - 1], label_output[i - 1]

        for i in range(1, depth):
            if i in grad_input:
                device = 'cuda:' + str(self.gpus[depth + (depth - i - 1)])
                if isinstance(grad_input[i], list):
                    grad_output[i - 1] = [tensor.to(device) for tensor in grad_input[i]]
                else:
                    grad_output[i - 1] = grad_input[i].to(device)
                grad_output_idx[i - 1] = grad_input_idx[i]
                del grad_input[i], grad_input_idx[i]

        # Apply each "asynchronous" block on the input, forward mode
        L = 0
        for i in range(depth):
            with torch.cuda.device(self.gpus[i]), torch.cuda.stream(self.streams[i]):
                if i in input:
                    if i < depth - 1:
                        output[i], label_output[i], output_idx[i] \
                            = modules[i].forward_with_decorations(input[i], label_input[i], input_idx[i])
                    else:
                        grad_input[depth - 1], meta = modules[i].forward_and_backward(input[i], label_input[i],
                                                                                      input_idx[i])
                    L, grad_input_idx[depth - 1], output[depth - 1], label_output[depth - 1] = meta

        # Apply each "asynchronous" block on the grad_output, backward mode -- except the last layer
        for i in range(depth - 1):
            with torch.cuda.device(self.gpus[depth + i]), torch.cuda.stream(self.streams[depth + i]):
                if i in grad_output:
                    grad_input[i], grad_input_idx[i] = modules[i].backward(grad_output[i], grad_output_idx[i])

        for stream in self.streams:
            stream.synchronize()

        # update
        for i in range(depth):
            modules[i].update()

        # If no output has been computed, handle it with a None.        
        if depth - 1 in output:
            return L, output[depth - 1], label_output[depth - 1]
        else:
            return L, None, None

    def _send_to(self, input, device):
        if isinstance(input, tuple):
            return input[0].to(device), input[1].to(device), input[2].to(device)
        else:
            return input.to(device)
