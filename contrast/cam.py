import torch
import sys
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2

sys.path.append('../')

from networks.resnet import resnet18

from utils.util import add_prefix, remove_prefix, mkdir


def load_pretrained_model(prefix):
    checkpoint = torch.load(add_prefix(prefix, 'model_best.pth.tar'))
    model = resnet18(is_ptrtrained=False)
    print('load pretrained resnet18 successfully.')
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    # print('best acc=%.4f' % checkpoint['best_accuracy'])
    return model


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


class CAM(object):
    def __init__(self, model, final_layer_name='layer4'):
        self.model = model
        self.final_layer_name = final_layer_name

        self.model.eval()

    def forward(self):
        pass

    def __call__(self, inputs):
        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        self.model._modules.get(self.final_layer_name).register_forward_hook(hook_feature)

        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())
        logit = self.model(inputs)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
        return CAMs[0]


def main(prefix, data_dir, saved_path):
    cam = CAM(model=load_pretrained_model(prefix))
    if data_dir == '../data/target_128':
        normalize = transforms.Normalize(mean=[0.651, 0.4391, 0.2991],
                                        std=[0.1046, 0.0846, 0.0611])
        print('load DR with size=128 successfully!')
    elif data_dir == '../data/split_contrast_dataset':
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
            parent_folder = '%s/%s' % (saved_path, phase)
            mkdir(parent_folder)
            cv2.imwrite('%s/%s' % (parent_folder, name), cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
    print(idx)
    return pred_results


if __name__ == '__main__':
    """
    usage:
    python cam.py ../classifier02 ../data/target128 ../cam01
    python cam.py ../classifier08 ../data/split_contrast_dataset ../cam02
    """
    prefix, data_dir, saved_path = sys.argv[1], sys.argv[2], sys.argv[3]
    main(prefix, data_dir, saved_path)
