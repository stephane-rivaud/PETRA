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

from dataset import get_dataset
from async_torch.layers.compression import get_quantizer, QuantizSimple
from async_torch.models.models_RevNet import make_layers_revnet_fixed_size
from async_torch.sequential_layers import AsynchronousSequential, SynchronousSequential, AsynchronousParallel
from scripts.utils import ProgressMeter, AverageMeter, accuracy, get_git_revision_hash, get_git_active_branch
from async_torch.optim.optimizer import add_optimizer
from async_torch.optim.scheduler import adjust_learning_rate_step_lr, adjust_learning_rate_polynomial, \
    adjust_learning_rate_cosine, adjust_learning_rate_goyal, adjust_learning_rate_goyal_original, warm_up_lr


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # ----- Dataset
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dir', default=None, type=str, help='path to dataset')

    # ----- Model
    parser.add_argument('--last-bn-zero-init', action='store_true', default=False,
                        help='Initialize gamma parameter of last bn to zero in each residual block.')
    parser.add_argument('--n-layers', default=20, type=int, help='Number of layers in the model')
    parser.add_argument('--hidden-size', default=256, type=int, help='Hidden size of the model')

    # ----- Optimization
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
    parser.add_argument('--lr-scaling', action='store_true', default=False,
                        help='Scale lr with accumulation steps.')
    parser.add_argument('--goyal-lr-scaling', action='store_true', default=False,
                        help='Use formula from Goyal for lr scaling (lr = k_workers * bs * lr / 256.'
                             'Precedence over --lr-scaling.')
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
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--remove-ctx-input', default=False, action='store_true',
                        help='Option to avoid storing input in buffer between the forward and backward pass.'
                             'Only effective for reversible architectures.')
    parser.add_argument('--remove-ctx-param', default=False, action='store_true',
                        help='Option to avoid storing weight in buffer between the forward and backward pass.')
    parser.add_argument('--approximate-input', default=False, action='store_true',
                        help='Option to avoid storing input in buffer between the forward and backward pass.'
                             'Only effective for reversible architectures.')
    parser.add_argument('--store-vjp', default=False, action='store_true',
                        help='Store the VJP in the buffer between the forward and backward pass.')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--accumulation-averaging', action='store_true', default=False,
                        help='Divide final loss by the number of accumulation steps')

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
    # wandb
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Use wandb for logging')
    parser.add_argument('--wandb-project', default='PETRA', type=str,
                        help='wandb project name')
    parser.add_argument('--wandb-entity', default='streethagore', type=str,
                        help='wandb entity')
    parser.add_argument('--wandb-offline', action='store_true', default=False,
                        help='Forces wandb into offline mode.')

    return parser.parse_args()


# Training
def train(epoch, dataloader, model, sync_period, device, dtype, opt):
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

        # original goyal scheduler
        if opt.scheduler == 'goyal_original':
            k_step = len(dataloader) * epoch + batch_idx
            n_step_tot = len(dataloader) * opt.max_epoch
            n_epoch_if_1_worker = opt.max_epoch
            lr = opt.lr
            world_size = opt.accumulation_steps
            batch_size = opt.batch_size
            dataset_name = opt.dataset
            lr = adjust_learning_rate_goyal_original(k_step, n_step_tot, n_epoch_if_1_worker, lr, world_size,
                                                     batch_size,
                                                     dataset_name)
            model.set_lr(lr)

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

        # forward backward and update
        with torch.no_grad():
            L, output, output_y = model.forward_and_update(inputs, targets, batch_idx)

        # synchronize forward and backward weights
        if batch_idx % sync_period == 0:
            model.synchronize_layers()

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

    return losses.avg, top1.avg, top5.avg


def test(epoch, dataloader, model, device, dtype, opt):
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
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # update meters
            batch_time.update(time.time() - end)
            losses.update(loss, inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    print(
        f'Test Epoch [{epoch}] | loss: {losses.avg:.4e} | test top1: {top1.avg:6.2f} | test top5: {top5.avg:6.2f} | learning rate: {model.modules[0].optimizers[0].lr:.4e}')
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(model, epoch, best_acc, best_epoch, run_id, filename='checkpoint.pth.tar'):
    state = model.state_list()
    cp = {'epoch': epoch, 'state_list': state, 'best_acc': best_acc, 'best_epoch': best_epoch, 'run_id': run_id}
    torch.save(cp, filename)


if __name__ == '__main__':
    import math

    parser_time = time.time()
    args = get_args()
    if not args.no_git:
        print('Active branch -->', get_git_active_branch())
        print('Commit ID -->', get_git_revision_hash())
    print(args)

    # load checkpoint and get run_id for wandb
    if args.resume:
        if os.path.isfile(args.resume):
            resume = True
            loc = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(args.resume, map_location=loc)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            if checkpoint['epoch'] >= args.max_epoch:
                print('Checkpoint is already at the last epoch. Exiting...')
                sys.exit(0)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            resume = False
    else:
        resume = False

    # wandb
    if args.use_wandb:
        if resume:
            run_id = checkpoint['run_id']
        else:
            run_id = wandb.util.generate_id()
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, resume=resume, id=run_id)
    else:
        run_id = None

    # Device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    dtype = torch.float32
    print('Device:', device, 'dtype:', dtype)

    # Data
    train_dataset, val_dataset, input_size, num_classes = get_dataset(args.dataset, args.dir)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # Model
    print('==> Building model..')
    quantizer = get_quantizer(args.quantizer)  # load the module using its name
    arch = make_layers_revnet_fixed_size(args.dataset, args.n_layers, num_classes, args.hidden_size,
                                         last_bn_zero_init=args.last_bn_zero_init,
                                         store_input=not args.remove_ctx_input, store_param=not args.remove_ctx_param,
                                         approximate_input=args.approximate_input, store_vjp=args.store_vjp,
                                         quantizer=quantizer, accumulation_steps=args.accumulation_steps,
                                         accumulation_averaging=args.accumulation_averaging)

    # Optimization
    if args.goyal_lr_scaling:
        args.lr = args.lr * args.accumulation_steps * args.batch_size / 256
    elif args.lr_scaling:
        args.lr = args.lr * args.accumulation_steps if args.lr_scaling else args.lr

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

    for layer in arch:
        if args.no_bn_weight_decay:
            # non-bias and non batch-norm parameters
            optim_kwargs['weight_decay'] = args.weight_decay
            condition = lambda name: '_bn' not in name and 'bias' not in name
            add_optimizer(layer, args.optimizer, optim_kwargs, condition)

            # bias and batch norm parameters
            optim_kwargs['weight_decay'] = 0.0
            condition = lambda name: '_bn' in name or 'bias' in name
            add_optimizer(layer, args.optimizer, optim_kwargs, condition)
        else:
            optim_kwargs['weight_decay'] = args.weight_decay
            add_optimizer(layer, args.optimizer, optim_kwargs)

    # Model container for training
    if args.synchronous:
        net = SynchronousSequential(arch).to(device, dtype)
    elif args.parallel:
        net = AsynchronousParallel(arch, n_devices=args.ngpus)
    else:
        net = AsynchronousSequential(arch).to(device, dtype)


    def count_parameters(model):
        num_param = 0
        for module in model.modules:
            for name in module.list_parameters:
                num_param += module.get_parameter(name).numel()
        return num_param


    print('Number of parameters:', count_parameters(net))

    # Scheduler
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

    # Resume options
    best_acc = 0  # best test accuracy
    best_epoch = 0  # best epoch
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if resume:
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        net.load_state_list(checkpoint['state_list'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    if args.use_wandb:
        wandb.summary.update({'start_epoch': start_epoch, 'modified_lr': args.lr})

    # Training
    train_time = time.time()
    for epoch in range(start_epoch, args.max_epoch):
        end = time.time()

        # adjust learning rate
        if not epoch < args.warm_up and args.scheduler not in ['goyal', 'goyal_original']:
            lr = adjust_learning_rate(epoch - args.warm_up)
            net.set_lr(lr)

        # train and test
        train_loss, train_top1, train_top5 = train(epoch, trainloader, net, args.synchronization_period, device, dtype,
                                                   args)
        test_loss, test_top1, test_top5 = test(epoch, testloader, net, device, dtype, args)
        if device == 'cuda':
            print(f'Maximum memory allocated {torch.cuda.max_memory_allocated(device=None)} bits')
            print(f'Maximum memory reserved {torch.cuda.max_memory_reserved(device=None)} bits')
            torch.cuda.reset_peak_memory_stats(device=None)

        # evaluation without compression
        if args.quantizer != 'QuantizSimple':
            net.apply(lambda m: setattr(m, 'quantizer', QuantizSimple()))
            print('...evaluation without compression')
            test_loss_no_comp, test_top1_no_comp, test_top5_no_comp = test(epoch, testloader, net, device, dtype, args)
            net.apply(lambda m: setattr(m, 'quantizer', quantizer()))

        print('epoch duration: {: .3f} secs'.format(time.time() - end))

        # store best accuracy
        if best_acc < test_top1:
            best_acc = test_top1
            best_epoch = epoch

        # log
        if args.use_wandb:
            log_dict = {
                'loss/train': train_loss, 'loss/test': test_loss,
                'top1/train': train_top1, 'top1/test': test_top1,
                'top5/train': train_top5, 'top5/test': test_top5,
                'learning_rate': net.modules[0].optimizers[0].lr
            }
            if args.quantizer != 'QuantizSimple':
                log_dict.update({'loss/test_no_compression': test_loss_no_comp,
                                 'top1/test_no_compression': test_top1_no_comp,
                                 'top5/test_no_compression': test_top5_no_comp})
            wandb.log(log_dict, step=epoch)
            wandb.summary.update({'best_epoch': best_epoch, 'best_acc': best_acc})

        # stop training if loss is NaN
        if math.isnan(test_loss):
            print('Loss is NaN, stopping training.')
            break

        # checkpointing
        if args.name_checkpoint:
            print('save checkpoint in ' + args.name_checkpoint)
            # we store the "NEXT" epoch to perform
            save_checkpoint(net, epoch + 1, best_acc, best_epoch, run_id, args.name_checkpoint)

    print('Training duration: {: .3f} secs'.format(time.time() - train_time))
    print('Total duration: {: .3f} secs'.format(time.time() - parser_time))
    if args.use_wandb:
        wandb.summary.update({'duration': time.time() - parser_time})
        run.finish()
