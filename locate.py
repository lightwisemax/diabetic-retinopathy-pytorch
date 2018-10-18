from __future__ import print_function

import os
import sys
import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.read_data import EasyDR

sys.path.append('./')
from utils.Logger import Logger
from models.unet import UNet
from models.resnet import resnet18
from models.Locator import Locator
from utils.util import set_prefix, write, add_prefix, to_np, to_variable, rgb2gray, weight_to_cpu

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on Diabetic Retinopathy Dataset')
parser.add_argument('--batch_size', '-b', default=90, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=90, type=int, help='training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lmbda', '-l', default=0.01, type=float, help='lmda constrain')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use gpu or not')
parser.add_argument('--step_size', default=40, type=int, help='learning rate decay interval')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay scope')
parser.add_argument('--interval_freq', '-i', default=12, type=int, help='printing log frequence')
parser.add_argument('--data', '-d', default='./data/easy_dr', help='path to dataset')
parser.add_argument('--co_power', '-k', default=2, type=int, help='power of gradient weight matrix')
parser.add_argument('--prefix', '-p', default='locate', type=str, help='folder prefix')
parser.add_argument('--pretrain_unet', default='./identical_mapping40/identical_mapping.pkl', type=str,
                    help='pretrained unet saved path')
parser.add_argument('--pretrain_classifier', default='./classifier05/model_best.pth.tar', type=str,
                    help='')
parser.add_argument('--is_pretrained_classifier', '-c', action='store_true', help='')


def main():
    global args, logger
    args = parser.parse_args()
    logger = Logger(add_prefix(args.prefix, 'logs'))
    set_prefix(args.prefix, __file__)
    model = model_builder()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # accelerate the speed of training
    cudnn.benchmark = True

    train_loader, val_loader = load_dataset()
    # class_names=['LESION', 'NORMAL']
    class_names = train_loader.dataset.class_names
    print(class_names)

    # learning rate decay per epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    since = time.time()
    print('-' * 10)
    for epoch in range(args.epochs):
        # adjust weight once unet can be nearly seen as an identical mapping
        exp_lr_scheduler.step()
        train(train_loader, model, optimizer, epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    validate(model, train_loader, val_loader)
    # save_typical_result(model)
    torch.save(model.state_dict(), add_prefix(args.prefix, 'locator.pkl'))
    write(vars(args), add_prefix(args.prefix, 'paras.txt'))


def load_dataset():
    global mean, std
    if args.data == './data/target_128':
        traindir = os.path.join(args.data, 'training')
        valdir = os.path.join(args.data, 'val')
        mean = [0.651, 0.4391, 0.2991]
        std = [0.1046, 0.0846, 0.0611]
        normalize = transforms.Normalize(mean, std)
        pre_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        ])
        post_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = EasyDR(traindir, pre_transforms, post_transforms, args.co_power)
        val_dataset = EasyDR(valdir, None, val_transforms, args.co_power)
        print('load targeted easy-classified diabetic retina dataset with size 128 to pretrain unet successfully!!')
    else:
        raise ValueError('')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True if args.cuda else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True if args.cuda else False)
    return train_loader, val_loader


def model_builder():
    classifier = resnet18(args.data, args.is_pretrained_classifier, args.pretrain_classifier)
    print('use resnet18')
    auto_encoder = UNet(3, depth=4, in_channels=3, has_conv1x1=True, has_bn=False)
    auto_encoder.load_state_dict(weight_to_cpu(args.pretrain_unet))
    print('load pretrained unet!')

    model = Locator(aer=auto_encoder, classifier=classifier)
    if args.cuda:
        model = DataParallel(model).cuda()
    else:
        raise ValueError('there is no gpu')

    return model


def restore(x):
    x = torch.squeeze(x)
    x = x.data.cpu()
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return x


def train(train_loader, model, optimizer, epoch):
    model.training(True)
    print('Epoch {}/{}'.format(epoch + 1, args.epochs))
    for idx, (inputs, labels, _, weights) in enumerate(train_loader):
        inputs = to_variable(inputs, args.cuda)
        labels = to_variable(labels, args.cuda)
        weights = to_variable(weights.unsqueeze(1), args.cuda)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        unet, outputs = model(inputs)
        u_loss = (weights * F.mse_loss(unet, inputs, reduce=False)).mean()
        error = F.cross_entropy(outputs, labels)
        loss = args.lmbda * u_loss + (1 - args.lmbda) * error
        loss.backward()
        optimizer.step()

        # save loss curve and learning rate
        step = epoch * len(train_loader.dataset) + idx
        info = {'unet_loss': u_loss.item(),
                'cross_entropy': error.item(),
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        if idx % args.interval_freq == 0:
            print('training unet_loss: {:.4f} cross_entropy_loss: {:.4f} loss:{:.4f}'.format(
                u_loss.item(), error.item(), loss.item()))


def validate(model, train_loader, val_loader):
    class_names = val_loader.dataset.class_names
    for phase in ['training', 'val']:
        for name in class_names:
            saved_path = '%s/%s/%s' % (args.prefix, phase, name.lower())
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
    model.eval()
    # save a sample from validate dataset
    phase = 'training'
    sample_inputs, sample_labels, sample_images_name, _ = next(iter(train_loader))
    sample_inputs = to_variable(sample_inputs, args.cuda)
    batch_size = sample_inputs.size(0)
    for idx in range(batch_size):
        single_image = sample_inputs[idx:(idx + 1), :, :, :]
        single_label = sample_labels[idx: idx + 1]
        single_name = sample_images_name[idx]
        single_unet, single_output = model(single_image)
        save_single_image(class_names[single_label.numpy()[0]].lower(),
                          single_name,
                          single_image,
                          single_output,
                          single_unet,
                          phase)
    # save training dataset
    phase = 'val'
    for idx, data in enumerate(val_loader):
        inputs, labels, name, _ = data
        # wrap them in Variable
        inputs = to_variable(inputs, args.cuda)
        unet, output = model(inputs)
        # save single image
        save_single_image(class_names[labels.numpy()[0]].lower(),
                          name[0],
                          inputs,
                          output,
                          unet,
                          phase)


def save_single_image(label, name, inputs, output, unet, phase):
    left = restore(inputs)
    right = restore(unet)

    # save network input image
    plt.figure(num='unet result', figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('source image: %s' % label)
    plt.imshow(left)
    plt.axis('off')
    predict = F.softmax(output, dim=1)
    predict = to_np(predict).flatten()

    # save network output image
    plt.subplot(2, 2, 2)
    plt.title('lesion: %.2f, normal: %.2f' % (predict[0], predict[1]))
    plt.imshow(right)
    plt.axis('off')

    # save difference directly
    diff = np.where(left > right, left - right, right - left).clip(0, 255)
    plt.subplot(2, 2, 3)
    plt.imshow(rgb2gray(diff), cmap='jet')
    plt.colorbar()
    plt.title('difference in abs gray')

    plt.tight_layout()
    plt.savefig(add_prefix(args.prefix, '%s/%s/%s' % (phase, label, name)))
    plt.close()


if __name__ == '__main__':
    main()
