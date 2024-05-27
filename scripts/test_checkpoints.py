import torch

from async_torch.models.utils import get_model
from async_torch.optim.optimizer import add_optimizer
from async_torch.sequential_layers.sequential import SynchronousSequential, AsynchronousSequential


# define architecture
model = get_model('cifar10', 'resnet18_2', False, None)

# add optimizer
optim_kwargs = {
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    }

for layer in model:
    add_optimizer(layer, 'sgd', optim_kwargs)

# wrap model into sequential container
model_sync = SynchronousSequential(model)
model_async = AsynchronousSequential(model)

# save checkpoints before training
print('Saving checkpoints before training...')
state_sync = model_sync.state_list()
torch.save(state_sync, '../state_sync.pth')

state_async = model_async.state_list()
torch.save(state_async, '../state_async.pth')

# train synchronous model and save checkpoint
print('Training synchronous model...')
x = torch.randn(7, 3, 32, 32)
y = torch.randint(0, 10, (7,))
idx = 0
model_sync.forward_and_update(x, y, idx)

print('Saving checkpoints after training...')
state_sync = model_sync.state_list()
torch.save(state_sync, '../state_sync_1_minibatch.pth')

# train asynchronous model and save checkpoint
print('Training asynchronous model...')
x = [torch.randn(7, 3, 32, 32) for _ in range(2*len(model))]
y = [torch.randint(0, 10, (7,)) for _ in range(2*len(model))]
idx = list(range(2*len(model)))

for x, y, idx in zip(x, y, idx):
    model_async.forward_and_update(x, y, idx)

    print('Saving checkpoints after training...')
    state_async = model_async.state_list()
    torch.save(state_async, f'state_async_{idx + 1}_minibatch.pth')
