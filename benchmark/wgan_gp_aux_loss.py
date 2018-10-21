import torch
from torch.autograd import grad
from torch import nn
from torch.optim import lr_scheduler

from models.unet import UNet
from u_d.base import base
from utils.util import append2xlsx, get_today, weight_to_cpu


class wgan_gp_aux_loss(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.aux_criterion = nn.NLLLoss().cuda()
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()



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
            fake_data_detached = fake_data.detach()

            real_dis_loss = -torch.mean(self.d(real_data)[0])
            fake_dis_loss = torch.mean(self.d(fake_data_detached)[0])

            theta = torch.rand((real_data.size(0), 1, 1, 1))
            if self.use_gpu:
                theta = theta.cuda()
            x_hat = theta * real_data.data + (1 - theta) * fake_data_detached.data
            x_hat.requires_grad = True
            pred_hat, _ = self.d(x_hat)
            if self.use_gpu:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = 10.0 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            dis_loss = real_dis_loss + fake_dis_loss + gradient_penalty
            dis_loss.backward()

            aux_input, aux_labels = torch.cat((fake_data_detached, real_data), 0), torch.cat((lesion_labels, normal_labels))
            shuffled_index = torch.randperm(aux_input.size(0)).cuda()
            aux_input, aux_labels = aux_input.index_select(0, shuffled_index), aux_labels.index_select(0, shuffled_index)
            aux_loss = self.aux_criterion(self.d(aux_input)[1], aux_labels)
            aux_loss.backward()

            # it's unnecessary to save gradient
            # self.save_gradient(epoch, idx)
            self.d_optimizer.step()

            if idx % self.n_update_gan == 0:
                self.u_optimizer.zero_grad()
                dis_output, aux_ouput = self.d(fake_data)
                d_loss_ = -self.alpha * torch.mean(dis_output)
                d_aux_loss = self.alpha * self.aux_criterion(aux_ouput, normal_labels)

                real_data_ = self.unet(real_data)
                u_loss_ = self.lmbda * (normal_gradient * self.l1_criterion(real_data_, real_data)).mean()
                loss = d_loss_ + d_aux_loss + u_loss_
                loss.backward()

                self.u_optimizer.step()

                w_distance = real_dis_loss.item() + fake_dis_loss.item()
                step = epoch * len(self.dataloader.dataset) + idx
                info = {'unet_loss': self.lmbda * u_loss_.item(),
                        'adversial_loss': self.alpha * d_loss_.item(),
                        'loss': self.alpha * d_loss_.item() + self.lmbda * u_loss_.item(),
                        'w_distance': w_distance,
                        'd_aux_loss': d_aux_loss.item(),
                        'lr': self.get_lr()}
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, step)

                if idx % self.interval == 0:
                    log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(d_aux_loss)+%.3f(gradient_penalty), ' \
                          'w_distance: %.3f, %.3f(u_d_loss)=%.3f(d_loss)+%.3f(aux_loss)+%.3f(l1_loss)' % (
                              epoch, self.epochs,
                              dis_loss.item() + aux_loss.item(),
                              real_dis_loss.item(),
                              fake_dis_loss.item(),
                              d_aux_loss.item(),
                              gradient_penalty.item(),
                              w_distance,
                              d_loss_.item() + d_aux_loss.item() + u_loss_.item(),
                              d_loss_.item(),
                              d_aux_loss.item(),
                              u_loss_.item())
                    print(log)
                    self.log_lst.append(log)

    def __dict__(self):
        attribute = self.attribute2dict()
        attribute['lmbda'] = self.lmbda
        attribute['alpha'] = self.alpha
        attribute['n_update_gan'] = self.n_update_gan
        return attribute

    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(0.0, 0.9))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(0.0, 0.9))
        self.u_lr_scheduler = lr_scheduler.ExponentialLR(self.u_optimizer, gamma=0.996)
        self.d_lr_scheduler = lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.996)

    def para2xlsx(self, path):
        prefix = self.prefix
        date = get_today()
        para = 'gpus={0},training_strategies={1},pretrained unet={2},' \
               'exponential_decay=0.996,batch_size={3},epochs={4},lmbda={5},alpha={6},' \
               'n_update_gan={7},lr={8},gan_type={9}(dowmsampling={10},depth={11})'.format(
            self.gpu_counts, self.training_strategies, self.is_pretrained_unet, self.batch_size, self.epochs,
            self.lmbda, self.alpha, self.n_update_gan, self.lr, self.gan_type, self.dowmsampling, self.d_depth
        )
        append2xlsx([[prefix, date, para]], path)

