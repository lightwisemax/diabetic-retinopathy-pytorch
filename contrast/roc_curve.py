import sys
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


sys.path.append('../')
from explore.save_all_results import evaluate
from utils.read_data import ConcatDataset
from utils.util import read, mkdir, add_prefix, remove_prefix, write
from contrast.cam import CAM
from contrast.models import vgg19
from networks.resnet import resnet18
from contrast.grad_cam import GradCam, preprocess_image
from contrast.grad_cam import select_visulization_nodes

plt.switch_backend('agg')


def save_data(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)


def read_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


class Our(object):
    """
    predict results using our proposed methods
    """

    def __init__(self, prefix, epoch, data_dir):
        self.prefix = prefix
        self.epoch = epoch
        self.data_dir = data_dir
        self.batch_size = 64

        self.groundtruth_dict = read(os.path.join(self.data_dir, 'groundtruth.txt'))
        self.auto_encoder = evaluate.get_unet('../%s/epoch_%s/g.pkl' % (prefix, epoch))
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.auto_encoder = DataParallel(self.auto_encoder)

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

    def __call__(self):
        y_true = []
        scores = []
        img_idx = 0
        for (lesion_data, _, lesion_names, _, _, _, _, _) in self.dataloader:
            if self.cuda:
                lesion_data = lesion_data.cuda()
            nums = min(self.batch_size, lesion_data.size(0))
            for idx in range(nums):
                single_image = lesion_data[idx:(idx + 1), :, :, :]
                single_name = lesion_names[idx]
                g, pred, = self.calculate(single_image, single_name)
                y_true.extend(g)
                scores.extend(pred)
                img_idx += 1
                if img_idx % 50 == 0:
                    print('[%d/%d]' % (img_idx, len(self.dataloader.dataset)))
            # break
        return scores, y_true

    @staticmethod
    def restore(x):
        x = x * 0.5 + 0.5
        x = torch.squeeze(x, 0)
        x = x.data.cpu().numpy().transpose((1, 2, 0))
        return x

    def calculate(self, image, name):

        left = self.restore(image)
        right = self.restore(self.auto_encoder(image))
        diff = np.abs(left-right)

        pred = np.clip((diff[:,:, 0]), 0.0, 1.0)
        # cv2.imwrite('test.png', np.uint8(diff[:, :, 0] * 255))

        g = np.zeros((128, 128)).astype(np.int64)
        bounding_box_lst = self.groundtruth_dict[name]
        for bounding_box in bounding_box_lst:
            pos_x, pos_y, size = bounding_box[0], bounding_box[1], bounding_box[2]
            g[pos_y: pos_y + size, pos_x: pos_x + size] = 1

        pred = pred - np.min(pred)
        pred = pred / np.max(pred)

        return g.reshape(-1).tolist(), pred.reshape(-1).tolist()


def plot_roc_curve(our_scores, our_true, cam_scores, cam_true, grad_scores, grad_true, saved_path):
    """
    references: https://github.com/JakenHerman/Plot_ROC_Curve/blob/master/roc_test.py
                https://blog.csdn.net/site1997/article/details/79180384
    """
    mkdir(saved_path)
    cam_fpr, cam_tpr, _ = roc_curve(cam_true, cam_scores)
    grad_fpr, grad_tpr, _ = roc_curve(grad_true, grad_scores)
    our_fpr, our_tpr, _ = roc_curve(our_true, our_scores)
    # plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.plot(cam_fpr, cam_tpr, color='green', label='cam', linewidth=1.0)
    plt.plot(grad_fpr, grad_tpr, color='red', label='grad-cam', linewidth=1.5)
    plt.plot(our_fpr, our_tpr, color='skyblue', label='our', linewidth=2.0)
    plt.legend()
    plt.savefig(os.path.join(saved_path, 'roc_curve.png'))
    plt.close()
    print('plot roc curve successfully.')

    cam_roc_auc = auc(cam_fpr, cam_tpr)
    grad_roc_auc = auc(grad_fpr, grad_tpr)
    our_roc_auc = auc(our_fpr, our_tpr)
    auc_scores = dict(cam_roc_auc=cam_roc_auc, grad_roc_auc=grad_roc_auc, our_roc_auc=our_roc_auc)
    print(auc_scores)
    write(auc_scores, '%s/auc_scores.txt' % saved_path)


def load_pretrained_model(prefix, model_type):
    if model_type == 'resnet':
        model = resnet18(is_ptrtrained=False)
    elif model_type == 'vgg':
        model = vgg19(num_classes=2, pretrained=False)
    else:
        raise ValueError('')

    checkpoint = torch.load(add_prefix(prefix, 'model_best.pth.tar'))
    print('load pretrained model successfully.')
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    print('best acc=%.4f' % checkpoint['best_accuracy'])
    return model


def cam_pred(prefix, data_dir):
    """
    """
    groundtruth_dict = read(os.path.join('../data/contrast_dataset', 'groundtruth.txt'))

    cam = CAM(model=load_pretrained_model(prefix, 'resnet'))
    if data_dir == '../data/split_contrast_dataset':
        normalize = transforms.Normalize(mean=[0.7432, 0.661, 0.6283],
                                         std=[0.0344, 0.0364, 0.0413])
        print('load custom-defined skin dataset successfully!!!')
    else:
        raise ValueError('')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    pred_results = []
    y_true = []
    idx = 0
    for phase in ['train', 'val']:
        path = os.path.join(data_dir, phase)
        for name in os.listdir(path):
            if 'lesion' not in name:
                continue
            abs_path = os.path.join(path, name)
            img_pil = Image.open(abs_path)
            img_tensor = preprocess(img_pil).unsqueeze(0)
            heatmap = cam(img_tensor)
            idx += 1
            if idx % 50 == 0:
                print('[%d/%d]' % (idx, len(os.listdir(path))))
            heatmap = np.float32(heatmap) / 255
            pred_results.extend(heatmap.reshape(-1))

            y_true.extend(get_true(groundtruth_dict, name).reshape(-1).tolist())

    print(idx)
    return pred_results, y_true


def grad_cam_pred(prefix, data_dir):
    groundtruth_dict = read(os.path.join('../data/contrast_dataset', 'groundtruth.txt'))
    grad_cam = GradCam(model=load_pretrained_model(prefix, 'vgg'), target_layer_names=select_visulization_nodes(data_dir))
    target_index = None
    pred_results = []
    y_true = []
    idx = 0
    for phase in ['train', 'val']:
        path = os.path.join(data_dir, phase)
        for name in os.listdir(path):
            if 'lesion' not in name:
                continue
            idx += 1
            image_path = os.path.join(path, name)
            img = cv2.imread(image_path, 1)
            img = np.float32(cv2.resize(img, (128, 128))) / 255
            input = preprocess_image(img, data_dir)
            mask = grad_cam(input, target_index)
            if idx % 50 == 0:
                print('[%d/%d]' % (idx, len(os.listdir(path))))
            pred_results.extend(mask.reshape(-1))
            y_true.extend(get_true(groundtruth_dict, name).reshape(-1).tolist())

    return pred_results, y_true


def get_true(groundtruth_dict, name):
    g = np.zeros((128, 128)).astype(np.int64)
    bounding_box_lst = groundtruth_dict[name]
    for bounding_box in bounding_box_lst:
        pos_x, pos_y, size = bounding_box[0], bounding_box[1], bounding_box[2]
        g[pos_y: pos_y + size, pos_x: pos_x + size] = 1
    return g


def debug(saved_path):
    def our():
        our_scores, our_true = Our(prefix='gan268', epoch='1099', data_dir='../data/contrast_dataset')()
        assert len(our_scores) == len(our_true)
        print('compte our proposed methods successfully.')
        our_fpr, our_tpr, _ = roc_curve(our_true, our_scores)
        save_data(our_true, os.path.join(saved_path, 'our_true.pkl'))
        save_data(our_scores, os.path.join(saved_path, 'our_scores.pkl'))
        our_roc_auc = auc(our_fpr, our_tpr)
        plt.title('ROC curve')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.plot(our_fpr, our_tpr, color='skyblue', label='our')
        plt.legend()
        plt.savefig(os.path.join(saved_path, 'roc_curve.png'))
        plt.close()
        print(our_roc_auc)

    def grad():
        grad_scores, grad_true = grad_cam_pred(prefix='../classifier07', data_dir='../data/split_contrast_dataset')
        assert len(grad_scores) == len(grad_true)
        print('compte grad-cam successfully.')
        save_data(grad_true, os.path.join(saved_path, 'grad_true.pkl'))
        save_data(grad_scores, os.path.join(saved_path, 'grad_scores.pkl'))
        grad_fpr, grad_tpr, _ = roc_curve(grad_true, grad_scores)
        grad_roc_auc = auc(grad_fpr, grad_tpr)
        plt.title('ROC curve')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.plot(grad_fpr, grad_tpr, color='skyblue', label='our')
        plt.legend()
        plt.savefig(os.path.join(saved_path, 'roc_curve.png'))
        plt.close()
        print(grad_roc_auc)

    grad()


def main(saved_path, reuse=False, save_pred=True):
    mkdir(saved_path)
    if reuse:
        grad_scores = read_data(os.path.join(saved_path, 'grad_scores.pkl'))
        grad_true = read_data(os.path.join(saved_path, 'grad_true.pkl'))

        cam_true = read_data(os.path.join(saved_path, 'cam_true.pkl'))
        cam_scores = read_data(os.path.join(saved_path, 'cam_scores.pkl'))

        our_true = read_data(os.path.join(saved_path, 'our_true.pkl'))
        our_scores = read_data(os.path.join(saved_path, 'our_scores.pkl'))

        print('load data successfully!')
    else:
        grad_scores, grad_true = grad_cam_pred(prefix='../classifier07', data_dir='../data/split_contrast_dataset')
        print('compte grad-cam successfully.')

        cam_scores, cam_true = cam_pred(prefix='../classifier08', data_dir='../data/split_contrast_dataset')
        print('compute cam successfully')

        our_scores, our_true = Our(prefix='gan268', epoch='1099', data_dir='../data/contrast_dataset')()
        print('compte our proposed methods successfully.')

    if save_pred:
        save_data(our_true, os.path.join(saved_path, 'our_true.pkl'))
        save_data(our_scores, os.path.join(saved_path, 'our_scores.pkl'))

        save_data(cam_true, os.path.join(saved_path, 'cam_true.pkl'))
        save_data(cam_scores, os.path.join(saved_path, 'cam_scores.pkl'))

        save_data(grad_true, os.path.join(saved_path, 'grad_true.pkl'))
        save_data(grad_scores, os.path.join(saved_path, 'grad_scores.pkl'))
        print('save data successfully.')

    plot_roc_curve(our_scores, our_true, cam_scores, cam_true, grad_scores, grad_true, saved_path)
    # debug(saved_path)


if __name__ == '__main__':
    """
    compute roc curve of cam, grad_cam and our proposed methods
    """
    main(saved_path='../roc_curve', reuse=True, save_pred=False)
