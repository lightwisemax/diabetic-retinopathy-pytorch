import os
import shutil
import sys
import argparse
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from torch.backends import cudnn
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


sys.path.append('./')
from networks.resnet import resnet18
from contrast.models import vgg

from utils.util import set_prefix, write, add_prefix, remove_prefix
from utils.read_data import EasyDR

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on Diabetic Retinopathy Dataset')
parser.add_argument('--batch_size', '-b', default=5, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=90, type=int, help='training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use gpu or not')
parser.add_argument('--step_size', default=50, type=int, help='learning rate decay interval')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay scope')
parser.add_argument('--interval_freq', '-i', default=12, type=int, help='printing log frequence')
parser.add_argument('--data', '-d', default='./data/target_128', help='path to dataset')
parser.add_argument('--prefix', '-p', default='classifier', type=str, help='folder prefix')
parser.add_argument('--best_model_path', default='model_best.pth.tar', help='best model saved path')
parser.add_argument('--model_type', '-m', default='vgg', type=str, help='classifier type', choices=['vgg', 'resnet18'])


min_loss = 100000.0
best_acc = 0.0

def main():
    global args, min_loss, best_acc
    args = parser.parse_args()
    device_counts = torch.cuda.device_count()
    print('there is %d gpus in usage' % (device_counts))
    # save source script
    set_prefix(args.prefix, __file__)
    model = model_selector(args.model_type)
    print(model)
    if args.cuda:
        model = DataParallel(model).cuda()
    else:
        raise RuntimeError('there is no gpu')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # accelerate the speed of training
    cudnn.benchmark = True

    train_loader, val_loader = load_dataset()
    # class_names=['LESION', 'NORMAL']
    class_names = train_loader.dataset.class_names
    print(class_names)
    criterion = nn.BCELoss().cuda()

    # learning rate decay per epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    since = time.time()
    print('-' * 10)
    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        train(train_loader, model, optimizer, criterion, epoch)
        cur_loss, cur_acc = validate(model, val_loader, criterion)
        is_best = cur_loss < min_loss
        best_loss = min(cur_loss, min_loss)
        if is_best:
            best_acc = cur_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model_type,
            'state_dict': model.state_dict(),
            'min_loss': best_loss,
            'acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    check_point = torch.load(add_prefix(args.prefix, args.best_model_path))
    print('min_loss=%.4f, best_acc=%.4f' %(check_point['min_loss'], check_point['acc']))
    write(vars(args), add_prefix(args.prefix, 'paras.txt'))


def model_selector(model_type):
    if model_type == 'vgg':
        model = vgg()
    elif model_type == 'resnet18':
        model = resnet18(is_ptrtrained=False)
    else:
        raise ValueError('')
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # save training state after each epoch
    torch.save(state, add_prefix(args.prefix, filename))
    if is_best:
        shutil.copyfile(add_prefix(args.prefix, filename),
                        add_prefix(args.prefix, args.best_model_path))


def load_dataset():
    if args.data == './data/flip':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
        # pre_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        # ])
        pre_transforms = transforms.Compose([
            transforms.RandomRotation(5),
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
        train_dataset = EasyDR(traindir, pre_transforms, post_transforms, alpha=0)
        val_dataset = EasyDR(valdir, None, val_transforms, alpha=0)
        print('load flipped DR  successfully!!!')
    else:
        raise ValueError("parameter 'data' that means path to dataset must be in "
                         "['./data/target_128', './data/flip']")

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True if args.cuda else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True if args.cuda else False)
    return train_loader, val_loader


def train(train_loader, model, optimizer, criterion, epoch, threshold=0.5):
    model.train(True)
    print('Epoch {}/{}'.format(epoch + 1, args.epochs))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for idx, (inputs, labels, _, _) in enumerate(train_loader):
        # wrap them in Variable
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.float().cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = F.sigmoid(model(inputs).squeeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if idx % args.interval_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, idx * len(inputs), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))

        pred = torch.where(outputs.data > threshold, torch.ones_like(outputs.data), torch.zeros_like(outputs.data)).long()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(pred == labels.data.long()).item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)

    print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def validate(model, val_loader, criterion, threshold=0.5):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, _, _ in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.float().cuda()
        outputs = F.sigmoid(model(data).squeeze(1))
        test_loss += criterion(outputs, target).item()
        # get the index of the max log-probability
        pred = torch.where(outputs.data > threshold, torch.ones_like(outputs.data), torch.zeros_like(outputs.data)).long()

        correct += torch.sum(pred == target.data.long()).item()

    test_loss /= len(val_loader.dataset)
    test_acc = 100. * correct / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), test_acc))
    return test_loss, test_acc


if __name__ == '__main__':
    main()
