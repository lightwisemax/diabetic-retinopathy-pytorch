import os
import time
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.discriminator import get_discriminator
from networks.unet import UNet
from utils.Logger import Logger
from utils.read_data import ConcatDataset
from utils.util import copy, add_prefix, weight_to_cpu, write, rgb2gray, write_list

plt.switch_backend('agg')


class base(object):
    def __init__(self, args):
        # initialize hyper-parameters
        self.data = args.data
        self.gan_type = args.gan_type
        self.d_depth = args.d_depth
        self.dowmsampling = args.dowmsampling
        self.gpu_counts = args.gpu_counts
        self.power = args.power
        self.batch_size = args.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.u_depth = args.u_depth
        self.is_pretrained_unet = args.is_pretrained_unet
        self.pretrain_unet_path = args.pretrain_unet_path

        self.lr = args.lr
        self.debug = args.debug
        self.prefix = args.prefix
        self.interval = args.interval
        self.n_update_gan = args.n_update_gan
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.beta1 = args.beta1

        self.training_strategies = args.training_strategies
        self.epoch_interval = 1 if self.debug else 50

        self.logger = Logger(add_prefix(self.prefix, 'tensorboard'))
        # normalize the images between [-1 and 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.dataloader = self.get_dataloader()
        self.d = get_discriminator(self.gan_type, self.d_depth, self.dowmsampling)
        self.unet = self.get_unet()

        self.log_lst = []

        if self.use_gpu:
            self.unet = DataParallel(self.unet).cuda()
            self.d = DataParallel(self.d).cuda()
        else:
            raise RuntimeWarning('there is no gpu available.')
        self.save_init_paras()
        self.get_optimizer()
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        write(vars(args), add_prefix(self.prefix, 'para.txt'))
        print('save hyperparameters successfully.')

    def get_lr(self):
        lr = []
        for param_group in self.d_optimizer.param_groups:
            lr += [param_group['lr']]
        return lr[0]

    def restore(self, x):
        x = torch.squeeze(x)
        x = x.data.cpu()
        for t, m, s in zip(x, self.mean, self.std):
            t.mul_(s).add_(m)
        x = x.numpy()
        x = np.transpose(x, (1, 2, 0))
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        return x

    def get_dataloader(self):
        if self.data == './data/gan':
            print('load DR with size 128 successfully!!')
        elif self.data == './data/gan_h_flip':
            print('load horizontal flipped DR with size 128 successfully!!')
        elif self.data == './data/gan1':
            print('load DR with distinct features!!')
        elif self.data == './data/gan3':
            print('load DR with 500 images.')
        elif self.data == './data/gan5':
            print('load DR with 500 images after preprocessing.')
        elif self.data == './data/gan7':
            print('load DR with images attaching ImageNet(lesion area size is equal to (32,32)).')
        elif self.data == './data/gan9':
            print('load resized skin dataset with random and tiny lesion area.')
        elif self.data == './data/gan11':
            print('load resizd skin dataset with one large lesion area.')
        elif self.data == './data/gan13':
            print('load DR with images attaching ImageNet(lesion area size is equal to (8,8)).')
        elif self.data == './data/gan15':
            print('attach 55 distinctly real lesion images based on gan13.')
        else:
            raise ValueError("the parameter data must be in ['./data/gan', './data/gan_h_flip']")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = ConcatDataset(data_dir=self.data,
                                transform=transform,
                                alpha=self.power
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 drop_last=True,
                                 pin_memory=True if self.use_gpu else False)
        return data_loader

    def get_unet(self):
        unet = UNet(3, depth=self.u_depth, in_channels=3)
        print(unet)
        print('load uent with depth %d and downsampling will be performed for %d times!!' % (self.u_depth, self.u_depth - 1))
        if self.is_pretrained_unet:
            unet.load_state_dict(weight_to_cpu(self.pretrain_unet_path))
            print('load pretrained unet')
        return unet

    def main(self):
        assert hasattr(self, 'u_lr_scheduler') and hasattr(self, 'd_lr_scheduler')
        print('training start!')
        start_time = time.time()
        print('d will be updated %d times while g will be updated for 1 time.' % self.n_update_gan)
        if self.interval % self.n_update_gan != 0:
            warnings.warn("It's hyperparameter n_update_gan is divisible by hyperparameter interval")
        for epoch in range(1, self.epochs + 1):
            self.u_lr_scheduler.step()
            self.d_lr_scheduler.step()

            self.train(epoch)
            if epoch % self.epoch_interval == 0:
                with torch.no_grad():
                    self.validate(epoch)
        with torch.no_grad():
            self.validate(self.epochs)

        total_ptime = time.time() - start_time
        if not self.debug:
            # note:relative path is based on the script u_d.py
            print('Training complete in {:.0f}m {:.0f}s'.format(
                total_ptime // 60, total_ptime % 60))

    def validate(self, epoch):
        """
        eval mode
        """
        real_data_score = []
        fake_data_score = []
        for i , (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in enumerate(self.dataloader):
            if i > 2:
                break
            if self.use_gpu:
                lesion_data, real_data = lesion_data.cuda(), real_data.cuda()
            phase = 'lesion_data'
            prefix_path = '%s/epoch_%d/%s' % (self.prefix, epoch, phase)
            lesion_output= self.d(self.unet(lesion_data))
            fake_data_score += list(lesion_output.squeeze().cpu().data.numpy().flatten())

            for idx in range(self.batch_size):
                single_image = lesion_data[idx:(idx + 1), :, :, :]
                single_name = lesion_names[idx]
                self.save_image(prefix_path, single_name, single_image)
                if self.debug:
                    break

            phase = 'normal_data'
            prefix_path = '%s/epoch_%d/%s' % (self.prefix, epoch, phase)
            normal_output = self.d(real_data)
            real_data_score += list(normal_output.squeeze().cpu().data.numpy().flatten())

            for idx in range(self.batch_size):
                single_image = real_data[idx:(idx + 1), :, :, :]
                single_name = normal_names[idx]
                self.save_image(prefix_path, single_name, single_image)
                if self.debug:
                    break

        prefix_path = '%s/epoch_%d' % (self.prefix, epoch)

        self.plot_hist('%s/score_distribution.png' % prefix_path, real_data_score, fake_data_score)
        torch.save(self.unet.state_dict(), add_prefix(prefix_path, 'g.pkl'))
        torch.save(self.d.state_dict(), add_prefix(prefix_path, 'd.pkl'))
        print('save model parameters successfully when epoch=%d' % epoch)

    def save_image(self, saved_path, name, inputs):
        """
        save unet output as a form of image
        """
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        output = self.unet(inputs)

        left = self.restore(inputs)
        right = self.restore(output)
        # The above two lines of code are wrong.To be precisely,errors will occur when the value of var left is less than
        # the value of var right.For example,left=217,right=220,then result is 253 after abs operation.
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

    def save_gradient(self, epoch, idx):
        """
        check bottom and top layer's gradient
        """
        if epoch % self.epoch_interval == 0 or epoch == 1:
            saved_path = '%s/gradient_epoch_%d' % (self.prefix, epoch)
            weights_top, weights_bottom = self.get_top_bottom_layer()
            weights_top, weights_bottom = list(weights_top.cpu().data.numpy().flatten()), list(
                weights_bottom.cpu().data.numpy().flatten())
            self.plot_gradient(saved_path, 'weights_top_%d.png' % idx, weights_top)
            self.plot_gradient(saved_path, 'weights_bottom_%d.png' % idx, weights_bottom)

    def plot_gradient(self, saved_path, phase, weights):
        """
        display gradient distribution in histogram
        """
        bins = np.linspace(min(weights), max(weights), 60)
        plt.hist(weights, bins=bins, alpha=0.3, label='gradient', edgecolor='k')
        plt.legend(loc='upper right')
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        plt.savefig('%s/%s' % (saved_path, phase))
        plt.close()

    def plot_hist(self, path, real_data, fake_data):
        bins = np.linspace(min(min(real_data), min(fake_data)), max(max(real_data), max(fake_data)), 60)
        plt.hist(real_data, bins=bins, alpha=0.3, label='real_score', edgecolor='k')
        plt.hist(fake_data, bins=bins, alpha=0.3, label='fake_score', edgecolor='k')
        plt.legend(loc='upper right')
        plt.savefig(path)
        plt.close()

    def get_top_bottom_layer(self):
        """
        save gradient of top and bottom layers to double-check and analyse
        """
        layer_names = list(dict(self.d.named_parameters()).keys())
        for name in layer_names:
            if 'conv' in name and 'weight' in name and 'bn' not in name and 'fc' not in name \
                    and not name.endswith('_u') and not name.endswith('_v'):
                bottom = name
                break
        for name in layer_names[::-1]:
            if 'conv' in name and 'weight' in name and 'bn' not in name and 'fc' not in name \
                    and not name.endswith('_u') and not name.endswith('_v'):
                top = name
                break
        return dict(self.d.named_parameters())[top].grad, dict(self.d.named_parameters())[bottom].grad

    def train(self, epoch):
        pass

    def get_optimizer(self):
        pass

    def save_running_script(self, script_path):
        """
        save the main running script to get differences between scripts
        """
        copy(script_path, add_prefix(self.prefix, script_path.split('/')[-1]))

    def save_log(self):
        write_list(self.log_lst, add_prefix(self.prefix, 'log.txt'))
        print('save running log successfully')

    def save_init_paras(self):
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        torch.save(self.unet.state_dict(), add_prefix(self.prefix, 'init_g_para.pkl'))
        torch.save(self.d.state_dict(), add_prefix(self.prefix, 'init_d_para.pkl'))
        print('save initial model parameters successfully')
