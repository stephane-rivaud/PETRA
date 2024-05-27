import torch
from torch.optim.adam import adam
from torch.optim.sgd import sgd


def add_optimizer(module, optimizer, optim_kwargs, condition=lambda x: True):
    list_parameters = [name for name in module.list_parameters if condition(name)]
    parameters = [getattr(module, name + '_backward') for name in list_parameters]

    if optimizer == 'sgd':
        optimizer_constructor = SGD
    elif optimizer == 'lars':
        optimizer_constructor = LARS
    elif optimizer == 'adam':
        optimizer_constructor = Adam
    else:
        raise ValueError(f'Wrong optimizer ({optimizer})')

    optimizer = optimizer_constructor(parameters, **optim_kwargs)
    optimizer.list_parameters = list_parameters
    module.optimizers.append(optimizer)


class SGD:
    def __init__(self, parameters, lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        self.list_momentum = [None] * len(parameters)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize

    def state_list(self):
        state = {
            'list_momentum': [momentum.clone() if momentum is not None else None for momentum in self.list_momentum],
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'maximize': self.maximize
        }
        return state

    def load_state_list(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def set_lr(self, lr):
        self.lr = lr

    def to(self, *args, **kwargs):
        for k, momentum in enumerate(self.list_momentum):
            if torch.is_tensor(momentum):
                self.list_momentum[k] = momentum.to(*args, **kwargs)

    def update(self, params, grads):
        sgd(params=params,
            d_p_list=grads,
            momentum_buffer_list=self.list_momentum,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            lr=self.lr,
            dampening=self.dampening,
            nesterov=self.nesterov,
            maximize=self.maximize)


class LARS:
    def __init__(self, parameters, lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.trust_coefficient = 0.02
        self.clip = True
        self.eps = 1e-8

        # momentum buffers
        self.list_momentum = [None] * len(parameters)

    def state_list(self):
        state = {
            'list_momentum': [momentum.clone() if momentum is not None else None for momentum in self.list_momentum],
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'maximize': self.maximize
        }
        return state

    def load_state_list(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def set_lr(self, lr):
        self.lr = lr

    def to(self, *args, **kwargs):
        for k, momentum in enumerate(self.list_momentum):
            if torch.is_tensor(momentum):
                self.list_momentum[k] = momentum.to(*args, **kwargs)

    def update(self, params, grads):
        with torch.no_grad():
            for p, grad in zip(params, grads):
                if grad is None:
                    continue
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad.data)

                if param_norm != 0 and grad_norm != 0:
                    # calculate adaptive lr + weight decay
                    adaptive_lr = self.trust_coefficient * (param_norm) / (
                            grad_norm + param_norm * self.weight_decay + self.eps)

                    # clip learning rate for LARC
                    if self.clip:
                        # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                        adaptive_lr = min(adaptive_lr / self.lr, 1)

                    grad.data += self.weight_decay * p.data
                    grad.data *= adaptive_lr

        sgd(params=params,
            d_p_list=grads,
            momentum_buffer_list=self.list_momentum,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=0.0,
            nesterov=self.nesterov,
            maximize=self.maximize)


class Adam:
    def __init__(self, parameters, lr, beta1, beta2, eps=1e-8, weight_decay=0, amsgrad=False, maximize=False):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.eps = eps

        # momentum buffers
        self.list_exp_avg = [torch.zeros_like(p) for p in parameters]
        self.list_exp_avg_sqs = [torch.zeros_like(p) for p in parameters]
        self.list_max_exp_avg_sqs = [torch.zeros_like(p) for p in parameters]
        self.list_state_steps = [torch.tensor(0.0) for _ in parameters]

    def state_list(self):
        state = {
            'list_exp_avg': [exp_avg.clone() if exp_avg is not None else None for exp_avg in self.list_exp_avg],
            'list_exp_avg_sqs': [exp_avg_sqs.clone() if exp_avg_sqs is not None else None for exp_avg_sqs in
                                 self.list_exp_avg_sqs],
            'list_max_exp_avg_sqs': [max_exp_avg_sqs.clone() if max_exp_avg_sqs is not None else None for
                                     max_exp_avg_sqs in self.list_max_exp_avg_sqs],
            'list_state_steps': [state_steps.clone() if state_steps is not None else None for state_steps in
                                 self.list_state_steps],
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            'maximize': self.maximize,
            'eps': self.eps
        }
        return state

    def load_state_list(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def set_lr(self, lr):
        self.lr = lr

    def to(self, *args, **kwargs):
        for k, (m0, m1, m2, s) in zip(self.list_exp_avg, self.list_exp_avg_sqs, self.list_max_exp_avg_sqs,
                                 self.list_state_steps):
            if torch.is_tensor(m0):
                self.list_exp_avg[k] = m0.to(*args, **kwargs)
            if torch.is_tensor(m1):
                self.list_exp_avg_sqs[k] = m1.to(*args, **kwargs)
            if torch.is_tensor(m2):
                self.list_max_exp_avg_sqs[k] = m2.to(*args, **kwargs)
            if torch.is_tensor(s):
                self.list_state_steps[k] = s.to(*args, **kwargs)

    def update(self, params, grads):
        adam(params=params,
             grads=grads,
             exp_avgs=self.list_exp_avg,
             exp_avg_sqs=self.list_exp_avg_sqs,
             max_exp_avg_sqs=self.list_max_exp_avg_sqs,
             state_steps=self.list_state_steps,
             amsgrad=self.amsgrad,
             beta1=self.beta1,
             beta2=self.beta2,
             lr=self.lr,
             weight_decay=self.weight_decay,
             eps=self.eps,
             maximize=self.maximize)
