"""
Grad-CAM can be widely used.
"""
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
import cv2

sys.path.append('../')
from networks.unet import UNet
from networks.resnet import resnet18
from utils.util import write, weight_to_cpu


class GradCam(object):
    def __init__(self, prefix, epoch):
        self.data_dir = '../data/gan'
        self.prefix = prefix
        self.epoch = epoch
        self.pretrained_classifier, self.pretrained_unet = self.load_pretrained_model()
        self.finalconv_name = 'layer4'
        self.saved_path = '../%s/grad_cam_%s' % (self.prefix, self.epoch)

    def load_pretrained_model(self):
        classifier, unet = resnet18(is_ptrtrained=False), UNet(3, depth=5, in_channels=3)
        print(classifier)
        print(unet)
        classifier.load_state_dict(weight_to_cpu(path='../%s/epoch_%s/c.pkl' % (self.prefix, self.epoch)))
        unet.load_state_dict(weight_to_cpu(path='../%s/epoch_%s/g.pkl' % (self.prefix, self.epoch)))
        print('load pretrained resnet18 successfully.')
        print('load pretrained unet successfully.')
        return classifier, unet

    @staticmethod
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 128x128
        size_upsample = (128, 128)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for _ in class_idx:
            cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def __call__(self):
        results = dict()
        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        self.pretrained_classifier._modules.get(self.finalconv_name).register_forward_hook(hook_feature)

        params = list(self.pretrained_classifier.parameters())

        weight_softmax = np.squeeze(params[-2].data.numpy())

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        classes = {0: 'lesion',
                   1: 'normal'}
        for phase in ['lesion', 'normal']:
            path = os.path.join(self.data_dir, phase)
            for name in os.listdir(path):
                abs_path = os.path.join(path, name)
                img_pil = Image.open(abs_path)
                img_tensor = preprocess(img_pil).unsqueeze(0)
                logit = self.pretrained_classifier(self.pretrained_unet(img_tensor) - img_tensor)
                h_x = F.softmax(logit, dim=1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                CAMs = self.returnCAM(features_blobs[0], weight_softmax, [idx[0]])
                print('predicted category=%s,probability=%.4f' % (classes[idx[0].item()], probs[0].item()))
                results = {'name': name,
                           'category': classes[idx[0].item()],
                           'probability': probs[0].item()
                           }
                img = cv2.imread(abs_path)
                height, width, _ = img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.2 + img * 0.6
                parent_folder = '%s/%s' % (self.saved_path, phase)

                if not os.path.exists(parent_folder):
                    os.makedirs(parent_folder)
                print('save %s to %s successfully.' % (name, os.path.join(parent_folder, name)))
                cv2.imwrite('%s/%s' % (parent_folder, name), result)
        write(results, '%s/results.json' % self.saved_path)


if __name__ == '__main__':
    """
    usage:
    python3 grad_cam.py gan156 499
    note: the frist parameter denotes results saved folder and the second parameter denotes saved folder.
    utilize grad-cam to visulize resnet18
    """
    # classifier, unet = resnet18(is_ptrtrained=False), UNet(3, depth=5, in_channels=3)

    prefix, epoch = sys.argv[1], sys.argv[2]
    GradCam(prefix, epoch)()