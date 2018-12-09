import torch
import cv2
import os
import sys
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

sys.path.append('../')
from contrast.models import vgg19
from utils.util import add_prefix, remove_prefix, mkdir, write


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img, data_dir):
    if data_dir == '../data/target_128':
        means=[0.651, 0.4391, 0.2991]
        stds=[0.1046, 0.0846, 0.0611]
    elif data_dir == '../data/split_contrast_dataset':
        means=[0.7432, 0.661, 0.6283]
        stds=[0.0344, 0.0364, 0.0413]
    else:
        raise ValueError('')

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, saved_path):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.5 * heatmap + 0.3 * np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(saved_path, np.uint8(255 * cam))


def load_pretrained_model(prefix):
    checkpoint = torch.load(add_prefix(prefix, 'model_best.pth.tar'))
    model = vgg19(num_classes=2, pretrained=False)
    print('load pretrained vgg19 successfully.')
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    print('best acc=%.4f' % checkpoint['best_accuracy'])
    return model


class GradCam(object):
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.classes = {
            0: 'lesion',
            1: 'normal'}

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (128, 128))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        logit = self.forward(input)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        # print('predicted category=%s,probability=%.4f' % (self.classes[idx[0].item()], probs[0].item()))
        return cam, dict(category=self.classes[idx[0].item()], probability=probs[0].item())


def main(prefix, data_dir, saved_path):
    grad_cam = GradCam(model=load_pretrained_model(prefix), target_layer_names=["35"])
    target_index = None
    idx = 0
    for phase in ['train', 'val']:
        path = os.path.join(data_dir, phase)
        parent_folder = '%s/%s' % (saved_path, phase)
        for name in os.listdir(path):
            if 'lesion' not in name:
                continue
            idx += 1
            image_path = os.path.join(path, name)
            img = cv2.imread(image_path, 1)
            img = np.float32(cv2.resize(img, (128, 128))) / 255
            input = preprocess_image(img, data_dir)
            mask, info = grad_cam(input, target_index)

            if idx % 50 == 0:
                print('[%d/%d]' % (idx, len(os.listdir(path))))
            parent_folder = '%s/%s' % (saved_path, phase)
            mkdir(parent_folder)
            show_cam_on_image(img, mask, '%s/%s' % (parent_folder, name))



if __name__ == '__main__':
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # reference: https://github.com/jacobgil/pytorch-grad-cam/blob/master/grad-cam.py
    """
    usage:
    python grad_cam.py ../classifier02 ../data/target128 ../grad_cam01
    python grad_cam.py ../classifier07 ../data/split_contrast_dataset ../grad_cam02
    """
    prefix, data_dir, saved_path = sys.argv[1], sys.argv[2], sys.argv[3]
    main(prefix, data_dir, saved_path)
