import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler

from contrast.models import vgg
from u_d.base import base
from utils.util import remove_prefix, add_prefix


class gan_c(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.mu = args.mu
        self.pretrained_epochs = args.pretrained_epochs
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()
        self.bce = nn.BCELoss()
        self.fix_classifier = self.get_classifier()
        print('discriminator will be updated for %d firstly.' % self.pretrained_epochs)

    def train(self, epoch):
        for idx, data in enumerate(self.dataloader, 1):
            lesion_data, _, _, _, real_data, normal_labels, _, normal_gradient = data
            if self.use_gpu:
                lesion_data, normal_gradient = lesion_data.cuda(), normal_gradient.unsqueeze(1).cuda()
                real_data, normal_labels = real_data.cuda(), normal_labels.float().cuda()
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
                lesion_l1_loss =  (self.l1_criterion(fake_data, lesion_data)).mean()
                c_loss = self.bce(F.sigmoid(self.fix_classifier(fake_data).squeeze(1)), normal_labels)

                u_loss = self.lmbda * normal_l1_loss + 0.1 * self.lmbda * lesion_l1_loss + self.alpha * d_loss_ + self.mu * c_loss
                u_loss.backward()

                self.u_optimizer.step()

                step = epoch * len(self.dataloader.dataset) + idx

                info = {'normal_l1_loss': self.lmbda * normal_l1_loss.item(),
                        'lesion_l1_loss': 0.1 * self.lmbda * lesion_l1_loss.item(),
                        'adversial_loss': self.alpha * d_loss_.item(),
                        'loss': u_loss.item(),
                        'w_distance': w_distance,
                        'lr': self.get_lr()}
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

                if idx % self.interval == 0:
                    log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty), ' \
                          'w_distance: %.3f, %.3f(u_d_loss)=%.3f(d_loss)+%.3f(normal_l1_loss)+%.3f(lesion_l1_loss)+%.3f(c_loss)' % (
                              epoch, self.epochs, d_loss.item(), d_real_loss.item(), d_fake_loss.item(),
                              gradient_penalty.item(), w_distance, u_loss.item(),
                              self.alpha * d_loss_.item(), self.lmbda * normal_l1_loss.item(), 0.1 * self.lmbda * lesion_l1_loss.item(), self.mu * c_loss.item())
                    print(log)
                    self.log_lst.append(log)

        if epoch == self.pretrained_epochs:
            with torch.no_grad():
                self.validate(epoch)

    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.u_lr_scheduler = lr_scheduler.ExponentialLR(self.u_optimizer, gamma=0.996)
        self.d_lr_scheduler = lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.996)

    def get_classifier(self, pretrained_path='./vgg02/'):
        checkpoint = torch.load(add_prefix(pretrained_path, 'model_best.pth.tar'))
        model = vgg()
        model.load_state_dict(remove_prefix(checkpoint['state_dict']))
        print(model)
        print('load pretrained vgg.')
        for param in model.parameters():
            param.requires_grad = False
        if self.use_gpu:
            model = DataParallel(model).cuda()
        else:
            raise RuntimeWarning('there is no gpu available.')
        return model