"""Train CIFAR10 with PyTorch."""

import os
import sys
import wandb

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import time
import argparse
from functools import partial

from scripts.dataset import get_dataset
from async_torch.layers.compression import get_quantizer
from async_torch.models.utils import get_model
from async_torch.sequential_layers import AsynchronousSequential, SynchronousSequential, AsynchronousParallel
from scripts.utils import ProgressMeter, AverageMeter, accuracy, get_git_revision_hash, get_git_active_branch
from async_torch.optim.optimizer import add_optimizer
from async_torch.optim.scheduler import adjust_learning_rate_step_lr, adjust_learning_rate_polynomial, \
    adjust_learning_rate_cosine, adjust_learning_rate_goyal, adjust_learning_rate_goyal_original, warm_up_lr


def compare_tensors(tensor1, tensor2, name_1=None, name_2=None, layer_index=None):
    all_close = torch.allclose(tensor1, tensor2)
    if not all_close:
        l_infty = (tensor1 - tensor2).abs().max().item()
        infty_norm_1 = tensor1.abs().max().item()
        infty_norm_2 = tensor2.abs().max().item()
        relative_infty_diff = 2 * l_infty / (infty_norm_1 + infty_norm_2)

        l2 = (tensor1 - tensor2).abs().mean().item()
        norm2_1 = tensor1.abs().mean().item()
        norm2_2 = tensor2.abs().mean().item()
        relative_l2_diff = 2 * l2 / (norm2_1 + norm2_2)

        if name_1 is not None and name_2 is not None:
            print(f'Layer {layer_index} | Error: {name_1} != {name_2}')
        print(f'L infinity difference: {l_infty: .4e} -- L inifinty norm: tensor1 {infty_norm_1: .4e}, tensor2 {infty_norm_2: .4e} -- relative difference: {relative_infty_diff: .4e}')
        print(f'L2 difference: {l2: .4e} -- L2 norm: tensor1 {norm2_1: .4e}, tensor2 {norm2_2: .4e} -- relative difference: {relative_l2_diff: .4e}')
    return all_close


def compare_models(model, model_vjp):
    # if isinstance(model, AsynchronousSequential) and isinstance(model_vjp, AsynchronousSequential):
    #     for (i, output), (i_vjp, output_vjp) in zip(model.output.items(), model_vjp.output.items()):
    #         assert compare_tensors(output, output_vjp, f'Output {i}', f'Output vjp {i}', i)
    #
    #     for (i, grad), (i_vjp, grad_vjp) in zip(model.grad_input.items(), model_vjp.grad_input.items()):
    #         assert compare_tensors(grad, grad_vjp, f'Grad input {i_vjp}', f'Grad input vjp {i_vjp}', i_vjp)

    for k, (module, module_vjp) in enumerate(zip(model.modules, model_vjp.modules)):
        for name, name_vjp in zip(module.list_parameters, module_vjp.list_parameters):
            assert name == name_vjp, f'Error: {name} != {name_vjp}'

            parameter = module.get_parameter(name, mode='forward')
            grad = module.get_gradient(name)

            parameter_vjp = module_vjp.get_parameter(name_vjp, mode='forward')
            grad_vjp = module_vjp.get_gradient(name_vjp)

            if grad is not None and grad_vjp is not None:
                assert compare_tensors(grad, grad_vjp, f'Grad {name}', f'Grad {name_vjp}', k)
            assert compare_tensors(parameter, parameter_vjp, name, name_vjp, k)

        for name, name_vjp in zip(module.list_buffers, module_vjp.list_buffers):
            assert name == name_vjp, f'Error: {name} != {name_vjp}'

            buffer = module.get_buffer(name, mode='forward')
            buffer_vjp = module_vjp.get_buffer(name_vjp, mode='forward')

            assert compare_tensors(buffer, buffer_vjp, name, name_vjp, k)


# Training
def train(epoch, dataloader, model, model_vjp, sync_period, device, dtype, opt):
    """Train for one epoch on the training set"""

    # meters
    print('\nEpoch: %d' % epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # training loop
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device, dtype), targets.to(device)
        data_time.update(time.time() - end)

        # Scheduler
        # warm-up
        if epoch < opt.warm_up and opt.scheduler not in ['goyal', 'goyal_original']:
            lr = warm_up_lr(epoch * len(dataloader) + batch_idx, opt.warm_up * len(dataloader), opt.lr)
            model.set_lr(lr)
            model_vjp.set_lr(lr)

        # original goyal scheduler
        if opt.scheduler == 'goyal_original':
            k_step = len(dataloader) * epoch + batch_idx
            n_step_tot = len(dataloader) * opt.max_epoch
            n_epoch_if_1_worker = opt.max_epoch
            lr = opt.lr
            world_size = opt.accumulation_steps
            batch_size = opt.batch_size
            dataset_name = opt.dataset
            lr = adjust_learning_rate_goyal_original(k_step, n_step_tot, n_epoch_if_1_worker, lr, world_size, batch_size,
                                                    dataset_name)
            model.set_lr(lr)
            model_vjp.set_lr(lr)

        # costumized goyal scheduler
        if opt.scheduler == 'goyal':
            k_step = len(dataloader) * epoch + batch_idx
            n_step_tot = len(dataloader) * opt.max_epoch
            n_epoch = opt.max_epoch
            milestones = opt.lr_decay_milestones
            lr = opt.lr
            world_size = opt.accumulation_steps
            batch_size = opt.batch_size
            lr = adjust_learning_rate_goyal(k_step, n_step_tot, n_epoch, milestones, lr, world_size, batch_size)
            model.set_lr(lr)
            model_vjp.set_lr(lr)

        # forward backward and update
        with torch.no_grad():
            L, output, output_y = model.forward_and_update(inputs, targets, batch_idx, set_grad_to_none=False)
            L_vjp, output_vjp, output_y_vjp = model_vjp.forward_and_update(inputs, targets, batch_idx, set_grad_to_none=False)

            if L > 0:
                assert compare_tensors(L, L_vjp, 'Loss', 'Loss_vjp')
                assert compare_tensors(output, output_vjp, 'Output', 'Output_vjp')
                assert compare_tensors(output_y, output_y_vjp, 'Output_y', 'Output_y_vjp')

        # checking that the forward weights are the same
        compare_models(model, model_vjp)

        # reset gradients
        model.apply(lambda module: module.set_grad_to_none())
        model_vjp.apply(lambda module: module.set_grad_to_none())

        # synchronize forward and backward weights
        if batch_idx % sync_period == 0:
            model.synchronize_layers()
            model_vjp.synchronize_layers()

        # update meters
        if output is not None:
            acc1, acc5 = accuracy(output, output_y, topk=(1, 5))
            losses.update(L, inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # display progress
        if batch_idx % opt.print_freq == 0:
            progress.display(batch_idx + 1)

    if opt.use_wandb:
        wandb.log({'loss/train': losses.avg, 'top1/train': top1.avg, 'top5/train': top5.avg}, commit=False)


def test(epoch, dataloader, model, model_vjp, device, dtype, opt):
    """Evaluate on the test set"""
    # meters
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.to(device)
    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device, dtype), targets.to(device)

            loss, outputs = model.forward(inputs, targets=targets)
            loss_vjp, outputs_vjp = model_vjp.forward(inputs, targets=targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc1_vjp, acc5_vjp = accuracy(outputs_vjp, targets, topk=(1, 5))

            assert torch.allclose(loss, loss_vjp), compare_tensors(loss, loss_vjp, 'Loss', 'Loss_vjp')
            assert torch.allclose(outputs, outputs_vjp), compare_tensors(outputs, outputs_vjp, 'Output', 'Output_vjp')
            assert torch.allclose(acc1, acc1_vjp), compare_tensors(acc1, acc1_vjp, 'Acc1', 'Acc1_vjp')
            assert torch.allclose(acc5, acc5_vjp), compare_tensors(acc5, acc5_vjp, 'Acc5', 'Acc5_vjp')

            # update meters
            batch_time.update(time.time() - end)
            losses.update(loss, inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    if opt.use_wandb:
        wandb.log({'loss/test': losses.avg, 'top1/test': top1.avg, 'top5/test': top5.avg}, step=epoch)

    print(
        f'Test Epoch [{epoch}] | loss: {losses.avg:.4e} | test top1: {top1.avg:6.2f} | test top5: {top5.avg:6.2f} | learning rate: {model.modules[0].optimizers[0].lr:.4e}')
    return losses.avg, top1.avg


def save_checkpoint(model, epoch, filename='checkpoint.pth.tar'):
    state = model.state_list()
    cp = {'epoch': epoch, 'state_list': state}
    torch.save(cp, filename)


if __name__ == '__main__':
    import math
    import random
    import numpy as np

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # ----- Dataset
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dir', default=None, type=str, help='path to dataset')

    # ----- Model
    parser.add_argument('--model', default='vgg', type=str, help='Architecture')
    parser.add_argument('--last-bn-zero-init', action='store_true', default=False,
                        help='Initialize gamma parameter of last bn to zero in each residual block.')

    # ----- Optimization
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch (useful on restarts)')
    parser.add_argument('--max-epoch', default=100, type=int, help='learning rate')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'lars', 'adam'],
                        help='define optimizer. supports lars, sgd and adam')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 1e-1)')
    # decay options
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--no-bn-weight-decay', action='store_true', default=False,
                        help='disable weight decay on batch-norm learnable parameters')
    # sgd and lars
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--dampening', type=float, default=0.0, help='dampening for SGD')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='Use nesterov momentum for SGD.')
    # adam
    parser.add_argument('--amsgrad', action='store_true', default=False, help='Use amsgrad in ADAM')
    parser.add_argument('--beta1', type=float, default=0.9, help='Momentum beta1 for ADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='Momentum beta2 for ADAM')

    # ----- Scheduler
    parser.add_argument('--warm-up', type=int, default=0, help='LR warm-up period.')
    parser.add_argument('--scheduler', type=str, default='steplr', help='LR Scheduler',
                        choices=['steplr', 'polynomial', 'cosine', 'goyal_original', 'goyal'])
    # steplr hyper-parameters
    parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[30, 60, 90],
                        help='decay learning rate at these milestone epochs (default: [30, 60, 90])')
    parser.add_argument('--lr-decay-fact', type=float, default=0.1,
                        help='learning rate decay factor to use at milestone epochs (default: 0.1)')

    # ----- Training
    parser.add_argument('--synchronous', default=False, action='store_true',
                        help='Use synchronous implementation')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='Use sequential asynchronous implementation')
    parser.add_argument('--quantizer', type=str, default='QuantizSimple', help='quantization class to use')
    parser.add_argument('--synchronization-period', type=int, default=1,
                        help='synchronization period (default: 1)')

    # ----- Gradient computation
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--remove_ctx_input', default=False, action='store_true',
                        help='Option to avoid storing input in buffer between the forward and backward pass.'
                             'Only effective for reversible architectures.')
    parser.add_argument('--remove_ctx_param', default=False, action='store_true',
                        help='Option to avoid storing weight in buffer between the forward and backward pass.')
    parser.add_argument('--store-vjp', default=False, action='store_true',
                        help='Store the VJP in the buffer between the forward and backward pass.')
    parser.add_argument('--accumulation-period', type=int, default=1,
                        help='accumulation period (default: 1)')
    # parser.add_argument('--compensate-delay', action='store_true', default=False, help='Delay compensation')

    # ----- Other Options
    # compute setup
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ngpus', default=1, type=int, help='number of GPUs')
    # logging
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    # checkpointing
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name-checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # git
    parser.add_argument('--no-git', action='store_true', default=False,
                        help='Prevent from accessing git informations.')
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Use wandb for logging')

    args = parser.parse_args()
    if not args.no_git:
        print('Active branch -->', get_git_active_branch())
        print('Commit ID -->', get_git_revision_hash())
    print(args)

    # Set seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    dtype = torch.float32
    print('Device:', device)

    # Data
    train_dataset, val_dataset, input_size, num_classes = get_dataset(args.dataset, args.dir)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # Model
    print('==> Building model..')
    quantizer = get_quantizer(args.quantizer)  # load the module using its name

    if args.accumulation_period == 1:
        size_averaging = None
    else:
        size_averaging = args.accumulation_period

    arch = get_model(args.dataset, args.model, args.last_bn_zero_init, store_input=not args.remove_ctx_input,
                     store_param=not args.remove_ctx_param, store_vjp=False, size_averaging=size_averaging,
                     quantizer=quantizer)

    arch_vjp = get_model(args.dataset, args.model, args.last_bn_zero_init, store_input=not args.remove_ctx_input,
                         store_param=not args.remove_ctx_param, store_vjp=True, size_averaging=size_averaging,
                         quantizer=quantizer)

    # Optimization
    if args.optimizer in ['sgd', 'lars']:
        optim_kwargs = {
            'lr': args.lr,
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov,
            'maximize': False
            }
    elif args.optimizer == 'adam':
        optim_kwargs = {
            'lr': args.lr,
            'amsgrad': False,
            'beta1': args.beta1,
            'beta2': args.beta2,
            'maximize': False
            }
    else:
        raise ValueError(f'Wrong optimizer ({args.optimizer})')

    for layer, layer_vjp in zip(arch, arch_vjp):
        if args.no_bn_weight_decay:
            # non batch-norm parameters
            optim_kwargs['weight_decay'] = args.weight_decay
            condition = lambda name: '_bn' not in name
            add_optimizer(layer, args.optimizer, optim_kwargs, condition)
            add_optimizer(layer_vjp, args.optimizer, optim_kwargs, condition)

            # bn parameters
            optim_kwargs['weight_decay'] = 0.0
            condition = lambda name: '_bn' in name
            add_optimizer(layer, args.optimizer, optim_kwargs, condition)
            add_optimizer(layer_vjp, args.optimizer, optim_kwargs, condition)
        else:
            optim_kwargs['weight_decay'] = args.weight_decay
            add_optimizer(layer, args.optimizer, optim_kwargs)
            add_optimizer(layer_vjp, args.optimizer, optim_kwargs)

    if args.synchronous:
        net = SynchronousSequential(arch).to(device, dtype)
        net_vjp = SynchronousSequential(arch_vjp).to(device, dtype)
    elif args.parallel:
        net = AsynchronousParallel(arch, n_devices=args.ngpus)
        net_vjp = AsynchronousParallel(arch_vjp, n_devices=args.ngpus)
    else:
        net = AsynchronousSequential(arch).to(device, dtype)
        net_vjp = AsynchronousSequential(arch_vjp).to(device, dtype)
    net_vjp.load_state_list(net.state_list())
    compare_models(net, net_vjp)

    if args.scheduler == 'steplr':
        lr_decay_milestones = [epoch - args.warm_up for epoch in args.lr_decay_milestones]
        adjust_learning_rate = partial(adjust_learning_rate_step_lr, milestones=lr_decay_milestones,
                                       lr_decay_fact=args.lr_decay_fact, lr_initial=args.lr)
    elif args.scheduler == 'polynomial':
        adjust_learning_rate = partial(adjust_learning_rate_polynomial, total_epochs=args.max_epoch - args.warm_up,
                                       lr_initial=args.lr)
    elif args.scheduler == 'cosine':
        adjust_learning_rate = partial(adjust_learning_rate_cosine, total_epochs=args.max_epoch - args.warm_up,
                                       lr_initial=args.lr)
    elif args.scheduler in ['goyal', 'goyal_original']:
        adjust_learning_rate = None
    else:
        raise ValueError(f'Wrong scheduler ({args.scheduler}).')

    # Training
    # resume options
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if torch.cuda.is_available():
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda'
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']œ
    #         net.load_state_list(checkpoint['state_list'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    if args.use_wandb:
        wandb.init(project='async-torch-lr-scheduling-search', entity=None, config=args)

    # training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        end = time.time()
        if not epoch < args.warm_up and args.scheduler not in ['goyal', 'goyal_original']:
            lr = adjust_learning_rate(epoch - args.warm_up)
            net.set_lr(lr)
            if args.use_wandb:
                wandb.log({'learning_rate': lr}, commit=False)
        train(epoch, trainloader, net, net_vjp, args.synchronization_period, device, dtype, args)
        test_loss, test_acc = test(epoch, testloader, net, net_vjp, device, dtype, args)

        # store best accuracy
        if best_acc < test_acc:
            best_acc = test_acc

        # checkpointing
        # if args.name_checkpoint:
        #     print('save checkpoint in ' + args.name_checkpoint)
        #     save_checkpoint(net, epoch + 1, args.name_checkpoint)  # we store the "NEXT" epoch to perform
        #
        # if args.quantizer != 'QuantizSimple':
        #     net.apply(lambda m: setattr(m, 'quantizer', QuantizSimple()))
        #     print('...evaluation without compression')
        #     test(epoch, testloader, net, device, args)
        #     net.apply(lambda m: setattr(m, 'quantizer', quantizer()))

        print('epoch duration: {: .3f} secs'.format(time.time() - end))

        if args.use_wandb:
            wandb.summary.update({'best_acc': best_acc})

        if math.isnan(test_loss):
            print('Loss is NaN, stopping training.')
            break

    if args.use_wandb:
        wandb.finish()
