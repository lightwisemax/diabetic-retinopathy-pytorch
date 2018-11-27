import time
import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler

from networks.discriminator import get_discriminator
from networks.unet import UNet
from u_d.base import base
from utils.Logger import Logger
from utils.piecewise_l1_loss import PiecewiseL1Loss
from utils.util import weight_to_cpu, add_prefix, read, write


class update_u(base):
    def __init__(self, args):
        """
        usage:
        python u_d.py --parent_folder_path=gan246 --load_epoch=1099 -d=./data/gan7 -ts=update_u
        """
        self.training_strategies = args.training_strategies
        self.parent_folder_path = args.parent_folder_path
        self.load_epoch = args.load_epoch
        self.epoch_interval = 3
        self.prefix = args.prefix
        self.data = args.data
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = args.batch_size
        self.config = self.load_config()
        self.epochs = args.epochs
        self.gamma = self.config['gamma']
        self.power = self.config['power']
        self.interval = args.interval
        self.debug = args.debug
        self.normal_weights = args.normal_weights
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
        # initialize exclusive attributes
        self.lmbda = self.config['lmbda']
        self.alpha = self.config['alpha']
        self.delta = self.config['delta']
        self.nums = self.config['nums']
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()
        if self.config['is_l1_loss']:
            self.lesion_criterion = nn.L1Loss().cuda()
            print('use L1Loss to restrain lesion data.')
        else:
            self.lesion_criterion = PiecewiseL1Loss(delta=self.delta, nums=self.nums).cuda()
            print('use PiecewiseL1Loss to restrain lesion data.')
        print('u will be updated while d is fixed.')
        self.get_optimizer()
        self.save_hyperparameters(args)

    def main(self):
        assert hasattr(self, 'u_lr_scheduler')
        print('training start!')
        start_time = time.time()
        print('only u will be updated while d is fixed.')

        for epoch in range(1, self.epochs + 1):
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
            lesion_data, _, _, _, real_data, _, _, normal_gradient = data
            if self.use_gpu:
                lesion_data, normal_gradient = lesion_data.cuda(), normal_gradient.unsqueeze(1).cuda()
                real_data = real_data.cuda()

            self.u_optimizer.zero_grad()
            fake_data = self.unet(lesion_data)
            dis_output = self.d(fake_data)
            d_loss_ = -torch.mean(dis_output)

            normal_l1_loss = (normal_gradient * self.l1_criterion(fake_data, real_data)).mean()
            lesion_l1_loss = self.lesion_criterion(fake_data, lesion_data)
            u_loss = self.lmbda * (self.normal_weights * normal_l1_loss + lesion_l1_loss) + self.alpha * d_loss_
            u_loss.backward()
            self.u_optimizer.step()

            if idx % self.interval == 0:
                step = epoch * len(self.dataloader.dataset) // self.batch_size + idx
                info = {'normal_l1_loss': self.lmbda * normal_l1_loss.item(),
                        'lesion_l1_loss': self.lmbda * lesion_l1_loss.item(),
                        'adversial_loss': self.alpha * d_loss_.item(),
                        'loss': u_loss.item(),
                        'lr': self.get_lr()}
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)
                log = '[%d/%d] %.3f(u_d_loss)=%.3f(d_loss)+%.3f(normal_l1_loss)+%.3f(lesion_l1_loss)' % (
                          epoch, self.epochs, u_loss.item(),
                          self.alpha * d_loss_.item(), self.lmbda * normal_l1_loss.item(),
                          self.lmbda * lesion_l1_loss.item())
                print(log)
                self.log_lst.append(log)

    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], 0.9))
        self.u_lr_scheduler = lr_scheduler.ExponentialLR(self.u_optimizer, gamma=1.0)

    def load_config(self):
        return read(add_prefix(self.parent_folder_path, 'para.txt'))

    def save_hyperparameters(self, args):
        self.config['parent_folder_path'] = self.parent_folder_path
        self.config['load_epoch'] = self.load_epoch
        self.config['epoch_interval'] = self.epoch_interval
        self.config['prefix'] = self.prefix
        self.config['data'] = self.data
        self.config['epochs'] = self.epochs
        self.config['training_strategies'] = self.training_strategies
        self.config['batch_size'] = self.batch_size
        write(self.config, add_prefix(self.prefix, 'para.txt'))
        print('save hyperparameters successfully.')

    def get_lr(self):
        lr = []
        for param_group in self.u_optimizer.param_groups:
            lr += [param_group['lr']]
        return lr[0]