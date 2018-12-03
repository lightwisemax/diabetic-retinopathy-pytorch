import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('../')
from explore.save_all_results import evaluate
from utils.read_data import ConcatDataset
from utils.util import read, rgb2gray, add_prefix, mkdir, write

plt.switch_backend('agg')

sys.path.append('../')


class DiceLoss(object):
    def __init__(self, prefix, epoch, data_dir):
        self.prefix = prefix
        self.epoch = epoch
        self.data_dir = data_dir
        self.batch_size = 64

        self.saved_path = '../%s/dice_loss%s/' % (self.prefix, self.epoch)
        mkdir(self.saved_path)
        self.groundtruth_dict = read(os.path.join(self.data_dir, 'groundtruth.txt'))
        self.auto_encoder = evaluate.get_unet('../%s/epoch_%s/g.pkl' % (prefix, epoch))
        self.dataloader = self.get_dataloader()

    def get_dataloader(self):
        if self.data_dir == '../data/contrast_dataset':
            print('load contrast dataset with size 128 successfully!!')
        else:
            raise ValueError("the parameter data must be in ['./data/contrast_dataset']")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = ConcatDataset(data_dir=self.data_dir,
                                transform=transform,
                                alpha=0
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=False,
                                 pin_memory=False)
        return data_loader

    def __call__(self, binary_thresh):
        results = []
        for (lesion_data, _, lesion_names, _, _, _, _, _) in self.dataloader:
            nums = min(self.batch_size, lesion_data.size(0))
            for idx in range(nums):
                single_image = lesion_data[idx:(idx + 1), :, :, :]
                single_name = lesion_names[idx]
                results.append(self.calculate(single_image, single_name, binary_thresh))
                break
        return round(sum(results) / len(results), 4)

    def calculate(self, image, name, binary_thresh):
        left = evaluate.restore(image)
        right = evaluate.restore(self.auto_encoder(image))
        diff = np.where(left > right, left - right, right - left).clip(0, 255)
        # binaryzation: background: 0 lesion areas: 1
        _, binary = cv2.threshold(rgb2gray(diff).astype(np.uint8), binary_thresh, 1, cv2.THRESH_BINARY)
        bounding_box_lst = self.groundtruth_dict[name]
        dice_loss_lst = []
        for bounding_box in bounding_box_lst:
            pos_x, pos_y, size = bounding_box[0], bounding_box[1], bounding_box[2]
            groundtruth = np.ones((size, size)).astype(np.uint8)
            pred = binary[pos_y: pos_y + size, pos_x: pos_x + size]
            dice_loss_lst.append(1 - distance.dice(groundtruth.reshape(-1), pred.reshape(-1)))

        return sum(dice_loss_lst) / len(dice_loss_lst)


def main(prefix, epoch, data_dir):
    saved_path = '../%s/dice_loss%s/' % (prefix, epoch)
    criterion = DiceLoss(prefix, epoch, data_dir)
    resutls = dict()
    # note the range of threshold: if the value is too small,the dice loss will be high but wrong because entire images will tend to 1.
    for thresh in range(1, 256):
        avg_dice_loss = criterion(thresh)
        resutls[thresh] = avg_dice_loss
        print('avg dice loss=%.4f,thresh=%d' % (avg_dice_loss, thresh))
    write(resutls, add_prefix(saved_path, 'results.txt'))


if __name__ == '__main__':
    """
    usage:
    python dice_loss.py gan262 499 ../data/contrast_dataset
    """
    prefix, epoch, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    main(prefix, epoch, data_dir)
