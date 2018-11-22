"""
save all images after model(i.e. u+d and u+d+c) is well-trained.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('../')
from networks.unet import UNet
from utils.read_data import ConcatDataset
from utils.util import weight_to_cpu, rgb2gray, add_prefix

plt.switch_backend('agg')


class evaluate(object):
    def __init__(self, prefix, epoch, data):
        """
        save all results
        :param prefix: parent folder such as 'gan145'
        :return:
        """
        self.batch_size = 64
        self.data = data
        self.prefix = prefix
        self.epoch = epoch
        self.auto_encoder = self.get_unet('../%s/epoch_%s/g.pkl' % (prefix, epoch))

    def save_single_image(self, saved_path, name, inputs):
        """
        save unet output as a form of image
        """
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        output = self.auto_encoder(inputs)

        left = self.restore(inputs)
        right = self.restore(output)

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
        print('file %s is saved to %s successfully.' %(name, add_prefix(saved_path, name)))
        plt.close()

    @staticmethod
    def restore(x):
        x = torch.squeeze(x, 0)
        x = x.data
        for t, m, s in zip(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
            t.mul_(s).add_(m)
        # transform Tensor to numpy
        x = x.numpy()
        x = np.transpose(x, (1, 2, 0))
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        return x

    @staticmethod
    def get_unet(pretrain_unet_path):
        unet = UNet(3, depth=5, in_channels=3)
        print(unet)
        print('load uent with depth %d and downsampling will be performed for 4 times!!')
        unet.load_state_dict(weight_to_cpu(pretrain_unet_path))
        print('load pretrained unet!')
        return unet

    def get_dataloader(self):
        if self.data == '../data/gan':
            print('load DR with size 128 successfully!!')
        elif self.data == '../data/gan_h_flip':
            print('load horizontal flipped DR with size 128 successfully!!')
        elif self.data == '../data/gan1':
            print('load DR with distinct features!!')
        elif self.data == '../data/gan3':
            print('load DR with 500 images.')
        elif self.data == '../data/gan5':
            print('load DR with 500 images after preprocessing.')
        elif self.data == '../data/gan7':
            print('load DR with images attaching ImageNet(lesion area size is equal to (32,32)).')
        elif self.data == '../data/gan9':
            print('load resized skin dataset with random and tiny lesion area.')
        elif self.data == '../data/gan11':
            print('load resizd skin dataset with one large lesion area.')
        elif self.data == '../data/gan13':
            print('load DR with images attaching ImageNet(lesion area size is equal to (8,8)).')
        elif self.data == '../data/gan15':
            print('attach 55 distinctly real lesion images based on gan13.')
        else:
            raise ValueError("the parameter data must be in ['./data/gan', './data/gan_h_flip']")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = ConcatDataset(data_dir=self.data,
                                transform=transform,
                                alpha=2
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 drop_last=True,
                                 pin_memory=False)
        return data_loader

    def __call__(self):
        dataloader = self.get_dataloader()
        for (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in dataloader:
            phase = 'lesion_data'
            prefix_path = '../%s/all_results_%s/%s' % (self.prefix, self.epoch, phase)
            for idx in range(self.batch_size):
                single_image = lesion_data[idx:(idx + 1), :, :, :]
                single_name = lesion_names[idx]
                self.save_single_image(prefix_path, single_name, single_image)

            phase = 'normal_data'
            prefix_path = '../%s/all_results_%s/%s' % (self.prefix, self.epoch, phase)
            for idx in range(self.batch_size):
                single_image = real_data[idx:(idx + 1), :, :, :]
                single_name = normal_names[idx]
                self.save_single_image(prefix_path, single_name, single_image)
        print('save all results completely.')


if __name__ == '__main__':
    """
    usage:
    python3 evaluate.py gan156 499 ../data/gan15
    note: the frist parameter denotes  parent folder and the second parameter denotes the status of model
    """
    prefix, epoch, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    evaluate(prefix, epoch, data_dir)()