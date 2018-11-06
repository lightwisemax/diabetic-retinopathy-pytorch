import torch.nn as nn


def initialize_weights(net):
    """
    initialize network
    note:It's different to initialize discriminator and classifier.
    For detail,please check the initialization of resnet and wgan-gp.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BatchNorm(nn.Module):
    """
    BatchBorm block: custom_conv-bn-leaky_relu where downsampling is performed if input channels is not equal to output channels。
    otherwise,the input size is kept.
    """
    def __init__(self, in_channels, out_channels):
        super(BatchNorm, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.drop_out = nn.Dropout2d(0.25)

    def forward(self, x):
        return self.drop_out(self.leaky_relu(self.bn(self.conv(x))))
        # return (self.leaky_relu(self.bn(self.conv(x))))




