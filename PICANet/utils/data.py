import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import cv2
import os
import os.path as osp
import sys
import random
import scipy.io as scio
import numpy as np

__all__ = ['SirstAugDataset', 'IRSTD1kDataset', 'NUDTSIRSTDataset', 'SirstDataset']

class SirstAugDataset(Data.Dataset):
    '''
    Return: Single channel, target=1, background=0
    '''
    def __init__(self, base_dir=r'datasets/sirst_aug',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.base_size = base_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.transform = augumentation()  # Renamed from tranform to transform for consistency

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img, mask = self.transform(img, mask)  # Apply data augmentation
        img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        img = img.reshape(1, self.base_size, self.base_size) / 255.0
        # Binarize mask: target=1, background=0
        mask = (mask > 0).astype(np.float32)
        mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask

    def __len__(self):
        return len(self.names)

class IRSTD1kDataset(Data.Dataset):
    '''
    Return: Single channel, target=1, background=0
    '''
    def __init__(self, base_dir=r'datasets/IRSTD-1k',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
                
        self.transform = augumentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        img = img.reshape(1, self.base_size, self.base_size) / 255.0
        # Binarize mask: target=1, background=0
        mask = (mask > 0).astype(np.float32)
        mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask

    def __len__(self):
        return len(self.names)

class NUDTSIRSTDataset(Data.Dataset):
    '''
    Return: Single channel, target=1, background=0
    '''
    def __init__(self, base_dir=r'datasets/NUDT-SIRST',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        img = img.reshape(1, self.base_size, self.base_size) / 255.0
        # Binarize mask: target=1, background=0
        mask = (mask > 0).astype(np.float32)
        mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask

    def __len__(self):
        return len(self.names)
        
class SirstDataset(Data.Dataset):
    def __init__(self, base_dir=r'datasets/SIRST',
                 mode='train', base_size=256):
        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        img = img.reshape(1, self.base_size, self.base_size) / 255.0
        # Binarize mask: target=1, background=0
        mask = (mask > 0).astype(np.float32)
        mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask



    def __len__(self):
        return len(self.names)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :].copy()
            target = target[::-1, :].copy()
        if random.random() < 0.5:
            input = input[:, ::-1].copy()
            target = target[:, ::-1].copy()
        if random.random() < 0.5:
            input = input.transpose(1, 0).copy()
            target = target.transpose(1, 0).copy()
        return input, target