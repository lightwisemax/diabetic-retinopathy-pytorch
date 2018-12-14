"""
usage:
python identical_mapping.py -b=64 -e=2 -p test -a=2 -i=20 -d=./data/easy_dr_128 -gpu=0,1,2,3 --debug
"""
import os
import sys
import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('./')
# from utils.Logger import Logger
from utils.read_data import ConcatDataset
from networks.unet import UNet
from utils.util import set_prefix, write, add_prefix, rgb2gray

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on Diabetic Retinopathy Dataset')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    help='batch size')
parser.add_argument('-e', '--epochs', default=120, type=int,
                    help='training epochs')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                    help='use gpu or not')
parser.add_argument('-i', '--interval_freq', default=12, type=int,
                    help='printing log frequence')
parser.add_argument('--power', '-k', type=int, default=2, help='power of weight')
parser.add_argument('-d', '--data', default='/data/zhangrong/gan',
                    help='path to dataset')
parser.add_argument('-p', '--prefix', required=True, type=str,
                    help='folder prefix')
parser.add_argument('--unet_depth', type=int, default=5, help='unet depth')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--local', action='store_true', default=False, help='data location')
parser.add_argument('--debug', action='store_true', default=False, help='in debug or not(default: false)')


def main():
    global args, logger
    args = parser.parse_args()
    # logger = Logger(add_prefix(args.prefix, 'logs'))
    set_prefix(args.prefix, __file__)
    model = UNet(3, depth=5, in_channels=3)
    print(model)
    print('load unet with depth=5')
    if args.cuda:
        model = DataParallel(model).cuda()
    else:
        raise RuntimeError('there is no gpu')
    criterion = nn.L1Loss(reduce=False).cuda()
    print('use l1_loss')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # accelerate the speed of training
    cudnn.benchmark = True

    data_loader = get_dataloader()
    # class_names=['LESION', 'NORMAL']
    # class_names = data_loader.dataset.class_names
    # print(class_names)

    since = time.time()
    print('-' * 10)
    for epoch in range(1, args.epochs + 1):
        train(data_loader, model, optimizer, criterion, epoch)
        if epoch % 40 == 0:
            validate(model, epoch, data_loader)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    validate(model, args.epochs, data_loader)
    # save model parameter
    torch.save(model.state_dict(), add_prefix(args.prefix, 'identical_mapping.pkl'))
    # save running parameter setting to json
    write(vars(args), add_prefix(args.prefix, 'paras.txt'))


def get_dataloader():
    if args.local:
        print('load data from local.')
        if args.data == '/data/zhangrong/gan':
            print('load DR with size 128 successfully!!')
        else:
            raise ValueError("the parameter data must be in ['/data/zhangrong/gan']")
    else:
        print('load data from data center.')
        if args.data == './data/gan':
            print('load DR with size 128 successfully!!')
        elif args.data == './data/contrast_dataset':
            print('load contrast dataset with size 128 successfully!!')
        else:
            raise ValueError("the parameter data must be in ['./data/gan']")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ConcatDataset(data_dir=args.data,
                            transform=transform,
                            alpha=args.power
                            )
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True,
                             pin_memory=True if args.cuda else False)
    return data_loader


def restore(x):
    x = x * 0.5 + 0.5
    x = torch.squeeze(x)
    x = x.data.cpu().numpy().transpose((1, 2, 0))
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return x


def train(data_loader, model, optimizer, criterion, epoch):
    model.train(True)
    print('Epoch {}/{}'.format(epoch, args.epochs))
    # Iterate over data.
    for idx, data in enumerate(data_loader, 1):
        lesion_data, _, _, lesion_gradient, _, _, _, _ = data
        if args.cuda:
            lesion_data, lesion_gradient = lesion_data.cuda(), lesion_gradient.unsqueeze(1).cuda()
        optimizer.zero_grad()
        # forward
        outputs = model(lesion_data)
        loss = (lesion_gradient * criterion(outputs, lesion_data)).mean()
        loss.backward()

        optimizer.step()
        step = epoch * int(len(data_loader.dataset) / args.batch_size) + idx
        info = {'loss': loss.item()}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step)
        if idx % args.interval_freq == 0:
            print('unet_loss: {:.4f}'.format(loss.item()))


def validate(model, epoch, data_loader):
    for i, (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in enumerate(data_loader):
        if i > 2 and epoch != args.epochs:
            break
        if args.cuda:
            lesion_data, real_data = lesion_data.cuda(), real_data.cuda()
        phase = 'lesion'
        prefix_path = '%s/epoch_%d/%s' % (args.prefix, epoch, phase)

        nums = min(args.batch_size, lesion_data.size(0))
        for idx in range(nums):
            single_image = lesion_data[idx:(idx + 1), :, :, :]
            single_name = lesion_names[idx]
            save_single_image(prefix_path, model, single_name, single_image)
            if args.debug:
                break

        phase = 'normal'
        prefix_path = '%s/epoch_%d/%s' % (args.prefix, epoch, phase)
        nums = min(args.batch_size, real_data.size(0))
        for idx in range(nums):
            single_image = real_data[idx:(idx + 1), :, :, :]
            single_name = normal_names[idx]
            save_single_image(prefix_path, model, single_name, single_image)
            if args.debug:
                break

    prefix_path = '%s/epoch_%d' % (args.prefix, epoch)
    torch.save(model.state_dict(), add_prefix(prefix_path, 'g.pkl'))
    print('save model parameters successfully when epoch=%d' % epoch)


def save_single_image(saved_path, model, name, inputs):
    """
    save unet output as a form of image
    """
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    output = model(inputs)

    left = restore(inputs)
    right = restore(output)

    diff = np.where(left > right, left - right, right - left).clip(0, 255).astype(np.uint8)
    plt.figure(num='unet result', figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('source image')
    plt.imshow(left)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title('unet output')
    plt.imshow(right)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(rgb2gray(diff), cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.title('difference in heatmap')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(rgb2gray(diff.clip(0, 32)), cmap='jet')
    plt.colorbar(orientation='horizontal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(add_prefix(saved_path, name))
    plt.close()


if __name__ == '__main__':
    main()
