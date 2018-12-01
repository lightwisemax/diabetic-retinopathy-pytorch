"""
https://www.jianshu.com/p/2fe73baa09b8
"""
import cv2
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('../')
from networks.discriminator import get_discriminator
from utils.read_data import ConcatDataset
from utils.util import weight_to_cpu


class FeatureVisualization(object):
    def __init__(self, prefix, epoch, selected_layer):
        # set random seed to ensure the same data in running
        torch.manual_seed(7)  # cpu
        np.random.seed(7)  # numpy
        random.seed(7)  # random and transforms

        self.prefix = prefix
        self.epoch = epoch
        self.data = '../data/gan_h_flip'
        self.batch_size = 64
        self.data_loader = self._get_dataloader()
        self.selected_layer = selected_layer
        self.pretrained_model = self._get_pretrained_model()

    def _get_dataloader(self):
        print('load horizontal flipped DR with size 128 successfully!!')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = ConcatDataset(data_dir=self.data,
                                transform=transform,
                                alpha=2
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=True,
                                 pin_memory=False)
        return data_loader

    def _get_pretrained_model(self):
        d = get_discriminator(dis_type='multi_scale', depth=7, dowmsampling=4)
        d.load_state_dict(weight_to_cpu(path='../%s/epoch_%s/d.pkl' % (self.prefix, self.epoch)))
        print('load pretrained d successfully.')
        return d


    def __call__(self):
        for idx, data in enumerate(self.data_loader, 1):
            lesion_data, _, lesion_names, _, _, _, _, _ = data

            features_blobs = []
            def hook_feature(module, input, output):
                features_blobs.append(output.data.numpy())
            sequential_name, layer_idx = self.selected_layer.split('.')[0], int(self.selected_layer.split('.')[1])
            self.pretrained_model._modules.get(sequential_name)[layer_idx].register_forward_hook(hook_feature)
            self.pretrained_model(lesion_data)
            # numpy
            features_blobs = features_blobs[0]
            features_blobs = 1.0 / (1 + np.exp(-1 * features_blobs))
            features_blobs = np.round(features_blobs * 255)
            for idx in range(self.batch_size):
                single_feature = (features_blobs[idx:(idx + 1), :, :, :]).squeeze()
                filters = single_feature.shape[0]
                for filter_idx in range(filters):
                    feature_map = single_feature[filter_idx: (filter_idx + 1), :, :].squeeze()
                    single_name = lesion_names[idx]
                    saved_path = '../%s/epoch_%s/feature_maps/%s/%s/' % (
                        self.prefix, self.epoch, single_name, self.selected_layer)
                    self._save(saved_path, '%d.png' %filter_idx, feature_map)
            break

    def _save(self, saved_path, name, feature_map):
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        if feature_map.shape[0] < 32:
            feature_map = cv2.resize(feature_map, (32, 32))
        cv2.imwrite(os.path.join(saved_path, name), feature_map)


if __name__ == '__main__':
    """
        usage:
        python3 feature_visulize.py  training_iterative01 400 'down_convs.0'
    """
    prefix, epoch, selected_layer = sys.argv[1], sys.argv[2], sys.argv[3]
    FeatureVisualization(prefix, epoch, selected_layer)()
