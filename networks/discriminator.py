"""
In this code,the difference between these classes is multiples of downsampling.
While-loop is strongly recommend.
reference: https://github.com/jaxony/unet-pytorch/blob/master/model.py
"""
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from networks.ops import BatchNorm, initialize_weights, BasicBlock
from networks.unet import conv1x1


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


class FPN(nn.Module):
    """
    reference: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
    """

    def __init__(self, block, layers, output_dim=1):
        self.inplanes = 64
        self.last_size = 32
        self.ouput_dim = output_dim

        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_size * self.last_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.ouput_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        print('the last feature size is (%d, %d)' % (self.last_size, self.last_size))

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

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p2 = self.smooth3(p2)
        # return p2, p3, p4, p5

        output = p2.view(-1, self.last_size * self.last_size)

        output = self.classifier(output)
        return output


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
        self.last_size = 128 // (2 ** self.downsampling)
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
        x = x.view(-1, 1)
        return x


def FPN18():
    return FPN(BasicBlock, [2, 2, 2, 2])


def get_discriminator(dis_type, depth, dowmsampling):
    if dis_type == 'conv_bn_leaky_relu':
        print('use conv_bn_leaky_relu as discriminator and downsampling will be achieved for %d times.' % dowmsampling)
        d = ConvBatchNormLeaky(depth, dowmsampling)
    elif dis_type == 'multi_scale':
        print('use MultiScale as discriminator')
        d = MultiScale(depth, dowmsampling)
    elif dis_type == 'resnet':
        print('use ResNet as discriminator')
        d = FPN18()
    elif dis_type == 'local_discriminator':
        d = LocalDiscriminator()
        print('use LocalDiscriminator as discriminator')
    else:
        raise ValueError("parameter discriminator type must be in ['conv_bn_leaky_relu', 'conv_leaky_relu']")
    print(d)
    print('use discriminator with depth of %d and last custom_conv feature size is (%d,%d)' % (
    d.depth, d.last_size, d.last_size))
    return d


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # d = ConvBatchNormLeaky(7, 4)
    # d = MultiScale(7, 4)
    # d = ResNet(BasicBlock, [2, 2, 2, 2])
    # d = LocalDiscriminator(downsampling=3)
    d = FPN18()
    print(d)
    tensor = torch.rand((2, 3, 128, 128))
    print(d(tensor))
