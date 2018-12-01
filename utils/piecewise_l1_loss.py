import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable


class PiecewiseL1Loss(nn.Module):
    def __init__(self, delta=20.0, nums=200.0 * 3, p=2, k=100.0):
        """
        piecewise l1_loss to make d more free
        :param delta: given any position (i,j),let d_i_j denotes differences between original image and corresponding output after UNet,we have:
            loss(d_i_j)=\sigma(d_i_j)=0 if d_i_j <= \delta,
            loss(d_i_j)=\sigma(d_i_j)= otherwise

        :param nums:
        :param p:
        """
        super(PiecewiseL1Loss, self).__init__()
        self.delta = delta
        if torch.cuda.is_available():
            self.nums = torch.ones(1).cuda() * nums
        else:
            self.nums = torch.ones(1) * nums
        self.p = p
        self.k = k

    @staticmethod
    def sigmoid(x, delta, alpha=0.4):
        return 1. / (1. + torch.exp(-alpha * (x - delta)))

    def forward(self, input, target):
        x = 127.5 * torch.abs(target - input)
        x = self.sigmoid(x, self.delta)
        changed_nums = torch.sum(F.relu(x)).unsqueeze(0)
        total_pixels = input.size(0) * input.size(2) * input.size(3)
        return self.k * torch.pow(F.relu(changed_nums - self.nums)/total_pixels, self.p)


def search():
    def sigmoid(x, alpha=0.0, delta=60):
        return 1. / (1. + np.exp(-alpha * (x - delta)))

    x = np.linspace(0, 50, 300)
    y1 = sigmoid(x, alpha=0.4)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y1, 'b-o', linewidth=1)
    plt.xlabel('intensify')
    plt.ylabel("loss")
    plt.title("sigma")
    plt.show()

    print('x=14,y=%.4f' % sigmoid(14, alpha=0.4))
    print('x=15,y=%.4f' % sigmoid(15, alpha=0.4))
    print('x=16,y=%.4f' % sigmoid(16, alpha=0.4))
    print('x=17,y=%.4f' % sigmoid(17, alpha=0.4))
    print('x=18,y=%.4f' % sigmoid(18, alpha=0.4))
    print('x=19,y=%.4f' % sigmoid(19, alpha=0.4))

    print('x=20,y=%.4f' % sigmoid(20, alpha=0.4))
    print('x=21,y=%.4f' % sigmoid(21, alpha=0.4))
    print('x=22,y=%.4f' % sigmoid(22, alpha=0.4))
    print('x=23,y=%.4f' % sigmoid(23, alpha=0.4))
    print('x=24,y=%.4f' % sigmoid(24, alpha=0.4))
    print('x=25,y=%.4f' % sigmoid(25, alpha=0.4))
    print('x=26,y=%.4f' % sigmoid(26, alpha=0.4))
    print('x=27,y=%.4f' % sigmoid(27, alpha=0.4))
    print('x=28,y=%.4f' % sigmoid(28, alpha=0.4))


def test_loss():
    import cv2
    x = cv2.imread('../data/gan_h_flip/lesion/lesion_16_right.jpeg')
    b, g, r = cv2.split(x)  # 拆分通道
    x = cv2.merge([r, g, b])  # 合并通道
    temp_y = x - 50
    x[:25, :25, :] = temp_y[:25, :25, :]
    x = x.astype('float32')
    x /= 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    x = Variable(torch.from_numpy(x).unsqueeze(0), requires_grad=True)
    print(x.size())

    y = cv2.imread('../data/gan_h_flip/lesion/lesion_16_right.jpeg')
    b, g, r = cv2.split(y)  # 拆分通道
    y = cv2.merge([r, g, b])  # 合并通道
    y = y.astype('float32')
    y /= 255.0
    y = (y - 0.5) / 0.5

    y = np.transpose(y, (2, 0, 1))
    y = Variable(torch.from_numpy(y).unsqueeze(0), requires_grad=True)
    criterion = PiecewiseL1Loss(nums=200.0 * 3)
    loss = criterion(x, y)
    print(loss.item())
    loss.backward()


if __name__ == '__main__':
    # search()
    test_loss()
