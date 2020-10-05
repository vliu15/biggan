# MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def collate_fn(batch):
    imgs, tgts = [], []
    for img, tgt in batch:
        imgs.append(img)
        tgts.append(tgt)

    return torch.stack(imgs, dim=0), torch.stack(tgts, dim=0)


class STL10(torchvision.datasets.STL10):
    ''' Custom STL10 dataset for BigGAN '''

    def __init__(self, *args, **kwargs):
        crop_size = kwargs.pop('crop_size', [64, 64])
        super().__init__(*args, **kwargs)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        tgt = torch.from_numpy(np.array(self.labels[idx])).long()
        img = torch.from_numpy(self.data[idx])
        img = self.transforms(img)
        return img, tgt

    def __len__(self):
        return len(self.data)


class ImageNet(torchvision.datasets.ImageNet):
    ''' Custom ImageNet dataset for BigGAN '''

    def __init__(self, *args, **kwargs):
        crop_size = kwargs.pop('crop_size', [256, 256])
        super().__init__(*args, **kwargs)

        self.transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        path, tgt = self.imgs[idx]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img, int(tgt)

    def __len__(self):
        return len(self.imgs)
