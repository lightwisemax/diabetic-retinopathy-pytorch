"""
In this code,the difference between these classes is multiples of downsampling.
While-loop is strongly recommend.
reference: https://github.com/jaxony/unet-pytorch/blob/master/model.py
"""
import math

import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.ops import BatchNorm, initialize_weights, BasicBlock
from networks.unet import conv1x1
from utils.read_data import EasyDR


class MultiScale(nn.Module):
    def __init__(self, depth, downsampling):
        super(MultiScale, self).__init__()
        self.start_filts = 64
        self.input_dim = 3
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling


        self.final_outs = self.start_filts
        down_conv = []
        down_conv.append(BatchNorm(self.input_dim, self.start_filts))

        ins = 64
        for i in range(1, self.downsampling):
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
            ins = self.outs
        self.final_outs += self.outs
        self.down_convs = nn.Sequential(*down_conv)

        convs = []
        for _ in range(self.depth - self.downsampling):
            convs.append(BatchNorm(self.outs, self.outs))
        self.final_outs += self.outs
        self.convs = nn.Sequential(*convs)
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)

        self.fc = nn.Sequential(
            nn.Linear(self.final_outs, self.output_dim)
        )
        initialize_weights(self)
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))

    @staticmethod
    def avg_pooling(feature):
        return F.avg_pool2d(feature, kernel_size=feature.size()[2:]).squeeze()

    def forward(self, x):
        for idx, module in enumerate(self.down_convs, 1):
            x = module(x)
            if idx == 1:
                feature1 = x
            if idx == self.downsampling:
                feature2 = x
        for module in self.convs:
            x = module(x)
        feature3 = x
        x = torch.cat((self.avg_pooling(feature1), self.avg_pooling(feature2), self.avg_pooling(feature3)), 1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 3
        self.depth = 4
        super(ResNet, self).__init__()
        self.last_size = 8

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = nn.AvgPool2d(self.last_size, stride=1)

        self.fc = nn.Linear(512, 1)

        initialize_weights(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class ConvBatchNormLeaky(nn.Module):
    """
    model architecture: downsampling*(custom_conv-bn-leaky_relu) + (depth-m)*(custom_conv-bn-leaky_relu)
    note in the former downsampling is performed after every custom_conv layer.And in the latter input size is invariable.
    Thus the parameter downsampling means the times of downsampling.
    """

    def __init__(self, depth, downsampling):
        """
        :param depth: downsampling 2^depth
        """
        super(ConvBatchNormLeaky, self).__init__()
        self.start_filts = 64
        self.input_dim = 3
        self.size = 128
        self.depth = depth
        self.output_dim = 1
        self.downsampling = downsampling

        assert self.depth >= self.downsampling
        down_conv = []
        for i in range(self.downsampling):
            ins = self.input_dim if i == 0 else self.outs
            self.outs = self.start_filts * (2 ** i)
            down_conv.append(BatchNorm(ins, self.outs))
        conv = []

        for _ in range(self.depth - self.downsampling):
            conv.append(BatchNorm(self.outs, self.outs))

        self.down_convs = nn.ModuleList(down_conv)
        self.conv = nn.ModuleList(conv)
        self.last_size = math.ceil(self.size / 2 ** self.downsampling)
        self.fc = nn.Sequential(nn.Linear(self.outs, self.output_dim))

        initialize_weights(self)
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))
        print('the downsmapling times is %d' % self.downsampling)

    def forward(self, x):
        for module in self.down_convs:
            x = module(x)
        for module in self.conv:
            x = module(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze()
        x = self.fc(x)
        return x


class LocalDiscriminator(nn.Module):
    def __init__(self, downsampling=3):
        super(LocalDiscriminator, self).__init__()
        self.downsampling = downsampling
        self.input_dim = 3
        self.start_filts = 64
        self.depth = self.downsampling
        self.last_size = 128 // (2**self.downsampling)
        convs = []

        for i in range(self.downsampling):
            ins = self.input_dim if i == 0 else self.outs
            self.outs = self.start_filts * (2 ** i)
            convs.append(BatchNorm(ins, self.outs))

        self.convs = nn.Sequential(*convs)
        self.final_conv = conv1x1(self.outs, 1)
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))
        initialize_weights(self)

    def forward(self, x):
        for module in self.convs:
            x = module(x)
        x = self.final_conv(x)
        x = x.view(-1,1)
        return x


def get_discriminator(dis_type, depth, dowmsampling):
    if dis_type == 'conv_bn_leaky_relu':
        print('use conv_bn_leaky_relu as discriminator and downsampling will be achieved for %d times.' % dowmsampling)
        d = ConvBatchNormLeaky(depth, dowmsampling)
    elif dis_type == 'multi_scale':
        print('use MultiScale as discriminator')
        d = MultiScale(depth, dowmsampling)
    elif dis_type == 'resnet':
        print('use ResNet as discriminator')
        d = ResNet(BasicBlock, [2, 2, 2, 2])
    elif dis_type == 'local_discriminator':
        d = LocalDiscriminator()
        print('use LocalDiscriminator as discriminator')
    else:
        raise ValueError("parameter discriminator type must be in ['conv_bn_leaky_relu', 'conv_leaky_relu']")
    print(d)
    print('use discriminator with depth of %d and last custom_conv feature size is (%d,%d)' % (d.depth, d.last_size, d.last_size))
    return d


def test_data_loader():
    global train_loader
    traindir = os.path.join('../data/target_128', 'train')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean, std)
    # pre_transforms: gradient will change in the same time as data augumentation goes.
    # post_transforms: ToTensor and Normalization
    pre_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
    ])
    post_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = EasyDR(traindir, pre_transforms, post_transforms, 2)
    train_loader = DataLoader(train_dataset,
                              batch_size=3,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=False)
    return train_loader


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_loader = test_data_loader()
    # d = ConvBatchNormLeaky(7, 4)
    # d = MultiScale(7, 4)
    # d = ResNet(BasicBlock, [2, 2, 2, 2])
    d = LocalDiscriminator(downsampling=3)

    print(d)
    tensor = torch.rand((2, 3, 128, 128))
    print(d(tensor).mean())