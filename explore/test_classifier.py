import torch
from torch.nn import DataParallel

from contrast.models import vgg
from networks.unet import UNet
from utils.util import weight_to_cpu, remove_prefix, add_prefix


def get_unet():
    unet = UNet(3, depth=5, in_channels=3)
    print(unet)
    unet.load_state_dict(weight_to_cpu('./gan202/epoch_1099'))
    print('load pretrained unet')
    return unet

model = get_unet()


def get_classifier(pretrained_path='./vgg02/'):
    checkpoint = torch.load(add_prefix(pretrained_path, 'model_best.pth.tar'))
    model = vgg()
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    print(model)
    print('load pretrained vgg.')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model




