import torch
import time
from torch.autograd import grad
from torch import nn

from u_d.base import base

class training_iterative(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()

    def main(self):
        print('training start!')
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            if epoch % 4 == 0:
                self.validate(epoch)
        self.validate(self.epochs)

        total_ptime = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            total_ptime // 60, total_ptime % 60))

    def train(self, epoch):
        for idx, data in enumerate(self.dataloader, 1):
            lesion_data, lesion_labels, _, _, real_data, normal_labels, _, normal_gradient = data
            if self.use_gpu:
                lesion_data, lesion_labels, normal_gradient = lesion_data.cuda(), lesion_labels.cuda(), normal_gradient.unsqueeze(
                    1).cuda()
                real_data, normal_labels = real_data.cuda(), normal_labels.cuda()

            if int(epoch / 20) % 2 == 0:
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
                w_distance = d_real_loss.item() + d_fake_loss.item()
                d_loss.backward()
                # it's unnecessary to save gradient
                # self.save_gradient(epoch, idx)
                self.d_optimizer.step()
                if idx % self.interval == 0:
                    log = '[%d/%d] %.3f=%.3f(d_real_loss)+%.3f(d_fake_loss)+%.3f(gradient_penalty), w_distance=%.3f' % \
                          (epoch, self.epochs, d_loss.item(), d_real_loss.item(), d_fake_loss.item(), gradient_penalty.item(), w_distance)
                    print(log)
                    self.log_lst.append(log)


            if int(epoch / 20) % 2 == 1:
                self.u_optimizer.zero_grad()
                fake_data = self.unet(lesion_data)
                dis_output = self.d(fake_data)
                d_loss_ = -torch.mean(dis_output)

                real_data_ = self.unet(real_data)
                u_loss_ = (normal_gradient * self.l1_criterion(real_data_, real_data)).mean()
                u_loss = self.lmbda * u_loss_ + self.alpha * d_loss_
                u_loss.backward()

                self.u_optimizer.step()

                if idx % self.interval == 0:
                    log = '[%d/%d] %.3f(u_d_loss)=%.3f(d_loss)+%.3f(l1_loss)' % (
                              epoch, self.epochs, u_loss.item(),
                              self.alpha * d_loss_.item(), self.lmbda * u_loss_.item())
                    print(log)
                    self.log_lst.append(log)


    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
