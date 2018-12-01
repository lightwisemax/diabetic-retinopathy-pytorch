"""
Read images and corresponding labels.
"""
import cv2
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from networks.unet import UNet
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.transforms import transforms

from utils.util import rgb2bgr, img2numpy


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, transform=None, alpha=6):
        """
        Args:
            data_dir: path to image directory.
            transform: optional transform to be applied on a sample.
        """
        assert alpha >= 0, 'the power parameter must be 0 at least!! '
        image_names = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_names = image_names
        self.transform = transform
        self.class_names = ['lesion', 'normal']
        self.alpha = alpha

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.data_dir, image_name)
        image = Image.open(path)
        # lesion: 0 normal: 1
        if 'lesion' in image_name:
            label = 0
        elif 'normal' in image_name:
            label = 1
        else:
            raise ValueError('')
        if self.transform is not None:
            image = self.transform(image)
        gradient = (self._get_gradient_magnitude(path) + 1.0) ** self.alpha
        return image, label, torch.from_numpy(gradient)

    def _get_gradient_magnitude(self, path):
        im = cv2.imread(path)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        mag = cv2.cvtColor(mag, cv2.COLOR_RGB2GRAY)
        mag = (mag / 255.0).astype('float32')
        # details enhancement
        return mag

    def __len__(self):
        return len(self.image_names)


class EasyDR(Dataset):
    """
    a easy-classified diabetic retina dataset with clearly feature for normal and lesion images respectively
    """

    def __init__(self, data_dir, pre_transform, post_transform, alpha=6):
        """
        Args:
            data_dir: path to image directory.
            pre_transform: data augumentation such as RandomHorizontalFlip
            post_transform: image preprocessing such as Normalization and ToTensor
            alpha: (1+w)^\alpha using power function or (1+alpha*w) using linear function
        """
        assert alpha >= 0, 'the power parameter must be 0 at least!! '
        image_names = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_names = image_names
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.class_names = ['lesion', 'normal']
        self.alpha = alpha

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.data_dir, image_name)
        image = Image.open(path)
        # lesion: 0 normal: 1
        if 'lesion' in image_name:
            label = 0
        elif 'normal' in image_name:
            label = 1
        else:
            raise ValueError('')
        if self.pre_transform is not None:
            image = self.pre_transform(image)
        gradient = (self._get_gradient_magnitude(image) + 1.0) ** self.alpha
        if self.post_transform is not None:
            image = self.post_transform(image)
        else:
            raise RuntimeError('')
        return image, label, image_name, torch.from_numpy(gradient)

    @staticmethod
    def _get_gradient_magnitude(im):
        # convert r-g-b channels to b-g-r channels because cv2.imread loads image in b-g-r channels
        im = rgb2bgr(img2numpy(im))
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        mag = cv2.cvtColor(mag, cv2.COLOR_RGB2GRAY)
        mag = (mag / 255.0).astype('float32')
        return mag

    def __len__(self):
        return len(self.image_names)


class ConcatDataset(Dataset):
    def __init__(self, data_dir, transform, alpha=6):
        """
        train simultaneously on two datasets
        Args:
            data_dir: path to image directory.
            alpha: (1+w)^\alpha using power function
        reference: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/16
        """
        assert alpha >= 0, 'the power parameter must be 0 at least!! '
        image_names = {x: os.listdir(os.path.join(data_dir, x)) for x in ['lesion', 'normal']}
        assert len(image_names['lesion']) == len(image_names['normal']), ''
        self.data_dir = data_dir
        self.image_names = image_names
        self.transform = transform
        self.alpha = alpha

    def __getitem__(self, index):
        lesion_image, lesion_label, lesion_name, lesion_gradient = self.__get_data(index, 'lesion')
        normal_image, normal_label, normal_name, normal_gradient = self.__get_data(index, 'normal')

        return lesion_image, lesion_label, lesion_name, lesion_gradient, normal_image, normal_label, normal_name, normal_gradient

    def __get_data(self, index, phase):
        name = self.image_names[phase][index]
        path = '%s/%s/%s' %(self.data_dir, phase, name)
        image = Image.open(path)
        gradient = (self._get_gradient_magnitude(image) + 1.0) ** self.alpha
        if self.transform is not None:
            image = self.transform(image)
        if 'lesion' in path:
            label = 0
        elif 'normal' in path:
            label = 1
        else:
            raise ValueError('')
        return image, label, name, torch.from_numpy(gradient)


    @staticmethod
    def _get_gradient_magnitude(im):
        # convert r-g-b channels to b-g-r channels because cv2.imread loads image in b-g-r channels
        im = rgb2bgr(img2numpy(im))
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        mag = cv2.cvtColor(mag, cv2.COLOR_RGB2GRAY)
        mag = (mag / 255.0).astype('float32')
        return mag

    def __len__(self):
        return len(self.image_names['lesion'])


def test_easy_dr():
    normalize = transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
    model = UNet(3, depth=3, in_channels=3)
    train_dir = '../data/gan/normal'
    train_dataset = EasyDR(data_dir=train_dir,
                           pre_transform=None,
                           post_transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize
                           ]),
                           )
    train_loader = DataLoader(dataset=train_dataset, batch_size=15,
                              shuffle=False, num_workers=2, pin_memory=False)
    for idx, (inputs, target, image_names, weight) in enumerate(train_loader):
        inputs = Variable(inputs)
        target = Variable(target)
        result = model(inputs)
        print(image_names)
        break

def test_concat_loader():
    data_dir = '../data/gan'
    ds = ConcatDataset(data_dir=data_dir,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5])
                                ])
                           )
    train_loader = DataLoader(dataset=ds, batch_size=15, shuffle=False, num_workers=2, pin_memory=False)
    for idx, data in enumerate(train_loader):
        print(data)
        break


if __name__ == '__main__':
    # test_easy_dr()
    test_concat_loader()