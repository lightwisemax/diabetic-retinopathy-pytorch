import sys
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


sys.path.append('../')
from networks.discriminator import get_discriminator
from networks.unet import UNet
from u_d.base import base
from utils.util import add_prefix, read, weight_to_cpu
from utils.read_data import ConcatDataset


class generate_scores(base):
    """
    usage:
    python generate_scores.py ../gan254 99 ../data/gan20 ../gan260
    note: the script runs in gpu environment.the script intends to test validate dataset if d is fully trained.
    """
    def __init__(self, prefix, epoch, data_dir, saved_path):
        self.prefix = prefix
        self.data = data_dir
        self.epoch = epoch
        self.batch_size = 32
        self.power = 2
        self.saved_path = saved_path
        self.dataloader = self.get_dataloader()
        self.config = self.load_config()
        self.unet = UNet(3, depth=self.config['u_depth'], in_channels=3)

        print(self.unet)
        print('load uent with depth %d and downsampling will be performed for %d times!!' % (
            self.config['u_depth'], self.config['u_depth'] - 1))
        self.unet.load_state_dict(weight_to_cpu('%s/epoch_%s/g.pkl' % (self.prefix, self.epoch)))
        print('load pretrained unet')

        self.d = get_discriminator(self.config['gan_type'], self.config['d_depth'], self.config['dowmsampling'])
        self.d.load_state_dict(weight_to_cpu('%s/epoch_%s/d.pkl' % (self.prefix, self.epoch)))
        print('load pretrained d')

    def __call__(self):
        real_data_score = []
        fake_data_score = []
        for i, (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in enumerate(self.dataloader):
            print('id=%d' %i)
            lesion_output = self.d(self.unet(lesion_data))
            fake_data_score += list(lesion_output.squeeze().data.numpy().flatten())
            normal_output = self.d(real_data)
            real_data_score += list(normal_output.squeeze().data.numpy().flatten())
        if not os.path.exists(self.saved_path):
            os.mkdir(self.saved_path)
        self.plot_hist('%s/score_distribution.png' % self.saved_path, real_data_score, fake_data_score)

    def load_config(self):
        return read(add_prefix(self.prefix, 'para.txt'))

    def get_dataloader(self):
        if self.data == '../data/gan17':
            print('training dataset of real normal data from gan15 in gan246.')
        elif self.data == '../data/gan18':
            print('validate dataset of unet lesion data output from gan246.')
        elif self.data == '../data/gan19':
            print('training dataset of real normal data from gan15 in gan253.')
        elif self.data == '../data/gan20':
            print('validate dataset of unet lesion data output from gan253.')
        else:
            raise ValueError("the parameter data must be in ['./data/gan', './data/gan_h_flip']")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = ConcatDataset(data_dir=self.data,
                                transform=transform,
                                alpha=self.power
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 drop_last=False,
                                 pin_memory=False)
        return data_loader


if __name__ == '__main__':
    prefix, epoch, data_dir, saved_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    assert len(sys.argv) == 5
    generate_scores(prefix, epoch, data_dir, saved_path)()



