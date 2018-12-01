"""
check whether generated image pixel is extremely easy to distinguish.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('../')
from benchmark.unet_generator import generator
from utils.read_data import EasyDR
from utils.util import weight_to_cpu

plt.switch_backend('agg')

def init():
    data = '../data/target_128'
    traindir = os.path.join(data, 'train')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
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
    train_dataset = EasyDR(traindir, pre_transforms, post_transforms, 2)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              pin_memory=False)
    unet = generator()
    unet.load_state_dict(weight_to_cpu('../gan112/epoch_400/g.pkl'))
    return unet, train_loader

def main():
    saved_path = '../analyse_gan112_generated_score'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    unet, train_loader = init()
    is_plot_fake = False
    is_plot_real = False
    real_data = []
    fake_data = []
    idx = 0
    for (inputs, labels, names, weights) in train_loader:
        if is_plot_fake and is_plot_real:
            plot_hist('%s/score_distribution_%d.png' % (saved_path, idx), real_data[idx], fake_data[idx])
            is_plot_fake = False
            is_plot_real = False
            idx += 1

        if labels.item() == 1 and not is_plot_real:
            real_data.append(inputs.squeeze().data.numpy().flatten())
            is_plot_real = True

        if labels.item() == 0 and not is_plot_fake:
            fake_data.append(unet(inputs).data.numpy().flatten())
            is_plot_fake = True


def plot_hist(path, real_data, fake_data):
    bins = np.linspace(min(min(real_data), min(fake_data)), max(max(real_data), max(fake_data)), 60)
    plt.hist(real_data, bins=bins, alpha=0.3, label='real_score', edgecolor='k')
    plt.hist(fake_data, bins=bins, alpha=0.3, label='fake_score', edgecolor='k')
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
