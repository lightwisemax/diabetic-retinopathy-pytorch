import torch
from torch.autograd import grad
from torch import nn
from torch.optim import lr_scheduler

from u_d.base import base
from utils.piecewise_l1_loss import PiecewiseL1Loss


class gan(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.delta = args.delta
        self.pretrained_epochs = args.pretrained_epochs
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()
        if args.is_l1_loss:
            self.lesion_criterion = nn.L1Loss().cuda()
            print('use L1Loss to restrain lesion data.')
        else:
            self.lesion_criterion = PiecewiseL1Loss(delta=self.delta).cuda()
            print('use PiecewiseL1Loss to restrain lesion data.')
        print('discriminator will be updated for %d firstly.' % self.pretrained_epochs)

    def train(self, epoch):
        for idx, data in enumerate(self.dataloader, 1):
            lesion_data, lesion_labels, _, _, real_data, normal_labels, _, normal_gradient = data
            if self.use_gpu:
                lesion_data, lesion_labels, normal_gradient = lesion_data.cuda(), lesion_labels.cuda(), normal_gradient.unsqueeze(
                    1).cuda()
                real_data, normal_labels = real_data.cuda(), normal_labels.cuda()
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
            w_distance = d_real_loss.item() + d_fake_loss.item()
            if epoch <= self.pretrained_epochs and idx % self.interval == 0:
                log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty),w_distance=%.3f' % (
                    epoch, self.epochs,
                    d_loss.item(), d_real_loss.item(), d_fake_loss.item(), gradient_penalty.item(), w_distance)
                print(log)
                self.log_lst.append(log)

            if epoch > self.pretrained_epochs and idx % self.n_update_gan == 0:
                self.u_optimizer.zero_grad()
                dis_output = self.d(fake_data)
                d_loss_ = -torch.mean(dis_output)

                real_data_ = self.unet(real_data)
                normal_l1_loss = (normal_gradient * self.l1_criterion(real_data_, real_data)).mean()
                lesion_l1_loss = self.lesion_criterion(fake_data, lesion_data)
                u_loss = self.lmbda * (normal_l1_loss + lesion_l1_loss) + self.alpha * d_loss_
                u_loss.backward()

                self.u_optimizer.step()

                step = epoch * len(self.dataloader.dataset)//self.batch_size + idx

                info = {'normal_l1_loss': self.lmbda * normal_l1_loss.item(),
                        'lesion_l1_loss': self.lmbda * lesion_l1_loss.item(),
                        'adversial_loss': self.alpha * d_loss_.item(),
                        'loss': u_loss.item(),
                        'w_distance': w_distance,
                        'lr': self.get_lr()}
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

                if idx % self.interval == 0:
                    log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty), ' \
                          'w_distance: %.3f, %.3f(u_d_loss)=%.3f(d_loss)+%.3f(normal_l1_loss)+%.3f(lesion_l1_loss)' % (
                              epoch, self.epochs, d_loss.item(), d_real_loss.item(), d_fake_loss.item(),
                              gradient_penalty.item(), w_distance, u_loss.item(),
                              self.alpha * d_loss_.item(), self.lmbda * normal_l1_loss.item(),
                              self.lmbda * lesion_l1_loss.item())
                    print(log)
                    self.log_lst.append(log)

        if epoch == self.pretrained_epochs:
            with torch.no_grad():
                self.validate(epoch)

    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.u_lr_scheduler = lr_scheduler.ExponentialLR(self.u_optimizer, gamma=1.0)
        self.d_lr_scheduler = lr_scheduler.ExponentialLR(self.d_optimizer, gamma=1.0)
