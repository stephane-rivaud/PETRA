import os

from torchvision import datasets, transforms

# Data
from scripts.imagenet32loader import ImageNetDownSample


def get_dataset(dataset, dir):
    print('==> Preparing data..')

    if dataset == 'cifar10':
        input_size = (3, 32, 32)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'cifar100':
        input_size = (3, 32, 32)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'imagenet32':
        input_size = (3, 32, 32)
        num_classes = 1000
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = ImageNetDownSample(root=ImageNetDownSample.data_dir, train=True, transform=transform_train)
        test_dataset = ImageNetDownSample(root=ImageNetDownSample.data_dir, train=False, transform=transform_test)

    elif dataset == 'imagenet':
        input_size = (3, 224, 224)
        num_classes = 1000

        # Data
        print('==> Preparing data..')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            os.path.join(dir, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = datasets.ImageFolder(
            os.path.join(dir, 'val'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise ValueError(f'Wrong dataset ({dataset}).')
    return train_dataset, test_dataset, input_size, num_classes
