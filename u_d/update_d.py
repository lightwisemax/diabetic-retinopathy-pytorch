import time
import os
import torch
from torch.autograd import grad
from torch.nn import DataParallel
from torch.optim import lr_scheduler

from networks.discriminator import get_discriminator
from networks.unet import UNet
from u_d.base import base
from utils.Logger import Logger
from utils.util import read, add_prefix, weight_to_cpu, write


class update_d(base):
    def __init__(self, args):
        """
        usage:
        python u_d.py --parent_folder_path=gan246 --load_epoch=1099 -d=./data/gan7 -ts=update_d
        note: don't call superclass
        """
        self.parent_folder_path = args.parent_folder_path
        self.load_epoch = args.load_epoch
        self.epoch_interval = 10
        self.config = self.load_config()
        self.power = self.config['power']
        self.batch_size = args.batch_size
        self.prefix = args.prefix
        self.data = args.data
        self.epochs = args.epochs
        self.gamma = self.config['gamma']
        self.interval = args.interval
        self.debug = args.debug
        self.use_gpu = torch.cuda.is_available()
        self.training_strategies = args.training_strategies
        self.log_lst = []

        self.unet = UNet(3, depth=self.config['u_depth'], in_channels=3)
        print(self.unet)
        print('load uent with depth %d and downsampling will be performed for %d times!!' % (
            self.config['u_depth'], self.config['u_depth'] - 1))
        self.unet.load_state_dict(weight_to_cpu('%s/epoch_%d/g.pkl' % (self.parent_folder_path, self.load_epoch)))
        print('load pretrained unet')

        self.d = get_discriminator(self.config['gan_type'], self.config['d_depth'], self.config['dowmsampling'])
        self.d.load_state_dict(weight_to_cpu('%s/epoch_%d/d.pkl' % (self.parent_folder_path, self.load_epoch)))
        print('load pretrained d')

        if torch.cuda.is_available():
            self.unet = DataParallel(self.unet).cuda()
            self.d = DataParallel(self.d).cuda()
        else:
            raise RuntimeWarning('there is no gpu available.')

        self.logger = Logger(add_prefix(self.prefix, 'tensorboard'))
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.dataloader = self.get_dataloader()
        self.get_optimizer()
        print('d will be updated while u is fixed.')
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        self.config['parent_folder_path'] = self.parent_folder_path
        self.config['load_epoch'] = self.load_epoch
        self.config['epoch_interval'] = self.epoch_interval
        self.config['prefix'] = self.prefix
        self.config['batch_size'] = self.batch_size
        self.config['data'] = self.data
        self.config['epochs'] = self.epochs
        self.config['training_strategies'] = self.training_strategies

        write(self.config, add_prefix(self.prefix, 'para.txt'))
        print('save hyperparameters successfully.')

    def load_config(self):
        return read(add_prefix(self.parent_folder_path, 'para.txt'))

    def main(self):
        assert hasattr(self, 'd_lr_scheduler')
        print('training start!')
        start_time = time.time()
        print('only d will be updated while u is fixed.')

        for epoch in range(1, self.epochs + 1):
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

    def train(self, epoch):
        for idx, data in enumerate(self.dataloader, 1):
            lesion_data, _, _, _, real_data, _, _, _ = data
            if self.use_gpu:
                lesion_data, real_data = lesion_data.cuda(), real_data.cuda()
            # training network: update d
            self.d_optimizer.zero_grad()
            fake_data = self.unet(lesion_data)

            real_dis_output = self.d(real_data)
            fake_dis_output = self.d(fake_data.detach())

            theta = torch.rand((real_data.size(0), 1, 1, 1))
            if self.use_gpu:
                theta = theta.cuda()
            x_hat = theta * real_data.data + (1 - theta) * fake_data.data
            x_hat.requires_grad = True
            pred_hat = self.d(x_hat)
            if self.use_gpu:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = self.gamma * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            d_real_loss = -torch.mean(real_dis_output)
            d_fake_loss = torch.mean(fake_dis_output)

            d_loss = d_real_loss + d_fake_loss + gradient_penalty
            d_loss.backward()

            # it's unnecessary to save gradient
            # self.save_gradient(epoch, idx)
            self.d_optimizer.step()
            if idx % self.interval == 0:
                w_distance = d_real_loss.item() + d_fake_loss.item()
                log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty),w_distance=%.3f' % (
                    epoch, self.epochs,
                    d_loss.item(), d_real_loss.item(), d_fake_loss.item(), gradient_penalty.item(), w_distance)
                print(log)
                self.log_lst.append(log)

                step = epoch * len(self.dataloader.dataset) // self.batch_size + idx

                info = {'w_distance': w_distance,
                        'lr': self.get_lr()
                        }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

    def get_optimizer(self):
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], 0.9))
        self.d_lr_scheduler = lr_scheduler.ExponentialLR(self.d_optimizer, gamma=1.0)

    def validate(self, epoch):
        """
        there is no need to save unet output
        """
        real_data_score = []
        fake_data_score = []
        for i , (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in enumerate(self.dataloader):
            if self.use_gpu:
                lesion_data, real_data = lesion_data.cuda(), real_data.cuda()
            lesion_output= self.d(self.unet(lesion_data))
            fake_data_score += list(lesion_output.squeeze().cpu().data.numpy().flatten())

            normal_output = self.d(real_data)
            real_data_score += list(normal_output.squeeze().cpu().data.numpy().flatten())

        prefix_path = '%s/epoch_%d' % (self.prefix, epoch)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        self.plot_hist('%s/score_distribution.png' % prefix_path, real_data_score, fake_data_score)
        torch.save(self.unet.state_dict(), add_prefix(prefix_path, 'g.pkl'))
        torch.save(self.d.state_dict(), add_prefix(prefix_path, 'd.pkl'))
        print('save model parameters successfully when epoch=%d' % epoch)

