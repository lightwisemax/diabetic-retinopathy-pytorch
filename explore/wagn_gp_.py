import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch.autograd import grad
from torch.optim import lr_scheduler

from u_d.base import base
from utils.util import add_prefix, write

plt.switch_backend('agg')


class wgan_gp_(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.min_distance = sys.float_info.max

    def train(self, epoch):
        for idx, (inputs, labels, _, _) in enumerate(self.lesion_loader, 1):
            # prepare data
            # prepare data for updating d
            real_data, lesion_data = self.extract_lesion_normal(inputs, labels)
            if self.use_gpu:
                real_data, lesion_data = real_data.cuda(), lesion_data.cuda()

            self.d_optimizer.zero_grad()
            fake_data = self.unet(lesion_data)
            # d differentiates healthy real data from unet healthy output and unet lesion output
            real_output, fake_output = self.d(real_data), self.d(fake_data.detach())
            d_real_loss, d_fake_loss = -torch.mean(real_output), torch.mean(fake_output)
            # gradient penalty
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
            gradient_penalty = 10.0 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            d_loss = d_real_loss + d_fake_loss + gradient_penalty
            w_distance = d_real_loss + d_fake_loss
            d_loss.backward()
            self.save_gradient(epoch, idx)
            self.d_optimizer.step()

            self.save_best_model(w_distance)
            if idx % self.interval == 0:
                step = epoch * len(self.lesion_loader.dataset) + idx
                info = {'loss': d_loss.item(),
                        'w_distance': w_distance.item(),
                        'lr': self.get_lr()}
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)
                print('[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty), w_distance: %.3f' % (
                    epoch, self.epochs,
                    d_loss.item(), d_real_loss.item(), d_fake_loss.item(), gradient_penalty.item(),
                    w_distance.item()))

    def save_best_model(self, distance):
        if distance.item() < self.min_distance:
            self.min_distance = distance
            torch.save({
                'state_dict': self.d.state_dict(),
                'min_distance': self.min_distance
            }, add_prefix(self.prefix, 'checkpoint.pth.tar'))

    def load_checkpoint(self):
        checkpoint = torch.load(add_prefix(self.prefix, 'checkpoint.pth.tar'))
        self.d.load_state_dict(checkpoint['state_dict'])
        print('min_distance=%.3f' % (checkpoint['min_distance']))

    def main(self):
        print('training start!')
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.d_lr_scheduler.step()
            self.train(epoch)
            if epoch % self.epoch_interval == 0:
                self.validate(epoch)
        self.load_checkpoint()
        self.validate(self.epochs)
        total_ptime = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            total_ptime // 60, total_ptime % 60))

        write(self.__dict__(), add_prefix(self.prefix, 'paras.txt'))
        # torch.save(self.d.state_dict(), add_prefix(self.prefix, 'd.pkl'))
        # print('save model parameters successfully')

    def validate(self, epoch):
        self.calculate_score('train', epoch)
        # self.calculate_score('val', epoch)
        print('save the score when epoch=%d' % epoch)

    def calculate_score(self, phase, epoch):
        if phase == 'train':
            samples, labels, _, _ = next(iter(self.lesion_loader))
        else:
            samples, labels, _, _ = next(iter(self.normal_loader))

        real_data, lesion_data = self.extract_lesion_normal(samples, labels)

        if self.use_gpu:
            real_data, lesion_data = real_data.cuda(), lesion_data.cuda()
        fake_data = self.unet(lesion_data)
        real_score, fake_score = self.d(real_data), self.d(fake_data.detach())
        real_score, fake_score = real_score.squeeze(1).cpu().data.numpy(), fake_score.squeeze(1).cpu().data.numpy()
        self.plot_histogram(real_score, fake_score, phase, epoch)
        # self.plot_histogram(, 'fake_data_score_epoch_%d' % epoch)

    def plot_histogram(self, real_score, fake_score, phase, epoch, group_nums=20):
        # print('real_score', list(real_score))
        # print('fake_score', list(fake_score))
        save_path = '%s/%s_epoch_%d' % (self.prefix, phase, epoch)
        bins = np.linspace(min(min(real_score), min(fake_score)), max(max(real_score), max(fake_score)), group_nums)
        plt.hist(real_score, bins=bins, alpha=0.3, label='real_score', edgecolor='k')
        plt.hist(fake_score, bins=bins, alpha=0.3, label='fake_score', edgecolor='k')
        plt.title('discriminator output score')
        plt.xlabel('discriminator output score')
        plt.ylabel('frequency')
        plt.legend(loc='upper right')

        plt.savefig(save_path)
        plt.close()
        print('plot histogram when epochs=%d' % epoch)


    def __dict__(self):
        return self.attribute2dict()

    def get_optimizer(self):
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(.0, 0.9))
        self.d_lr_scheduler = lr_scheduler.StepLR(self.d_optimizer, step_size=self.step_size, gamma=0.1)