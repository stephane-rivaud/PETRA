"""Create tools for Pytorch to load Downsampled ImageNet (32X32,64X64)

Thanks to the cifar.py provided by Pytorch.

Author: *****
Date:   Apr/21/2019

Data Preparation:
    1. Download unsampled data from ImageNet website.
    2. Unzip file  to rootPath. eg: /home/*****/Datasets/ImageNet64(no train, val folders)

Remark:
This tool is able to automatic recognize downsampled size.


Use this tool like cifar10 in datsets/torchvision.
"""


from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class ImageNetDownSample(data.Dataset):
    """`DownsampleImageNet`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]
    data_dir = 'data/imagenet32'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not check_imagenet32():
            prepare_imagenet32()

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def prepare_imagenet32():
    import urllib
    import zipfile

    # # Define data directory
    data_dir = "data/imagenet32"
    os.makedirs(data_dir, exist_ok=True)

    # Download zip file
    url = "https://figshare.com/ndownloader/articles/4959980/versions/2"
    local_filename = os.path.join(data_dir, 'imagenet32.zip')
    print(f'Downloading {url} to {local_filename}')
    urllib.request.urlretrieve(url, local_filename)
    print(f'\nDownload completed: {local_filename}')

    # Path to the zip file
    zip_file_path = local_filename

    # Directory to extract the contents to
    extract_to_dir = data_dir

    # Ensure the extract directory exists
    os.makedirs(extract_to_dir, exist_ok=True)

    # Open the zip file
    print(f'Extracting {zip_file_path} to {extract_to_dir}')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents
        zip_ref.extractall(extract_to_dir)

    print(f'Files extracted to: {extract_to_dir}')

    # Delete zip file
    os.remove(local_filename)


def check_imagenet32():
    # Check that data directory exists
    data_dir = "data/imagenet32"
    if not os.path.isdir(data_dir):
        return False
    # Check training data
    for split in range(1, 11):
        batch_split_path = os.path.join(data_dir, f'train_data_batch_{split}')
        if not os.path.isfile(batch_split_path):
            return False
    # Check validation data
    val_split_path = os.path.join(data_dir, 'val_data')
    if not os.path.isfile(val_split_path):
        return False
    return True


if __name__ == '__main__':
    if not check_imagenet32():
        prepare_imagenet32()
    else:
        print('ImageNet32 dataset is already there.')
