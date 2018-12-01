"""
CAM is only applicable to a particular kind of CNN architectures performing global average pooling over convolutional maps immediately prior to prediction
(i.e. conv feature maps → global average pooling → softmax layer).
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
from contrast.models import vgg
from networks.resnet import resnet18
from utils.util import add_prefix, remove_prefix, write


def load_pretrained_model(prefix, model_type):
    checkpoint = torch.load(add_prefix(prefix, 'model_best.pth.tar'))
    if model_type == 'vgg':
        model = vgg()
        print('load vgg successfully.')
    elif model_type == 'resnet':
        model = resnet18(is_ptrtrained=False)
        print('load resnet18 successfully.')
    else:
        raise ValueError('')
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    return model


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


def main(prefix, saved_path, model_type, data_dir='../data/target_128'):
    if model_type == 'vgg':
        finalconv_name = 'conv5'
    elif model_type == 'resnet':
        finalconv_name = 'layer4'
    else:
        raise ValueError('')
    results = dict()
    vgg_cam = load_pretrained_model(prefix, model_type)
    vgg_cam.eval()
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    vgg_cam._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(vgg_cam.parameters())

    weight_softmax = np.squeeze(params[-2 if model_type=='resnet' else -1].data.numpy())

    normalize = transforms.Normalize(mean=[0.651, 0.4391, 0.2991],
                                     std=[0.1046, 0.0846, 0.0611])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    classes = {0: 'lesion',
               1: 'normal'}
    for phase in ['train', 'val']:
        path = os.path.join(data_dir, phase)
        for name in os.listdir(path):
            abs_path = os.path.join(path, name)
            img_pil = Image.open(abs_path)
            img_tensor = preprocess(img_pil).unsqueeze(0)
            logit = vgg_cam(img_tensor)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            # print(features_blobs[0].shape)
            # print(weight_softmax)
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            print('predicted category=%s,probability=%.4f' % (classes[idx[0].item()], probs[0].item()))
            results[name] = {
                'category': classes[idx[0].item()],
                'probability': probs[0].item()
                }
            img = cv2.imread(abs_path)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.2 + img * 0.6
            parent_folder = '%s/%s' % (saved_path, phase)
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            print('save %s to %s successfully.' % (name, os.path.join(parent_folder, name)))
            cv2.imwrite('%s/%s' % (parent_folder, name), result)
    write(results, '%s/results.json' % saved_path)

if __name__ == '__main__':
    """
    usage:
    python3 cam.py ../vgg01 ../cam01
    note: the frist parameter denotes  classifier saved folder and the second parameter denotes saved folder
    """
    prefix, saved_path, model_type = sys.argv[1], sys.argv[2], sys.argv[3]
    main(prefix, saved_path, model_type)
