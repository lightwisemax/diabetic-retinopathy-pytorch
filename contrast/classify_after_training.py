"""
compute the score and categories
"""
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

sys.path.append('../')
from contrast.models import vgg19
from networks.resnet import resnet18
from utils.util import add_prefix, remove_prefix, write


def load_pretrained_model(pretrained_path, model_type):
    checkpoint = torch.load(add_prefix(pretrained_path, 'model_best.pth.tar'))
    if model_type == 'vgg':
        model = vgg19(pretrained=False, num_classes=2)
        print('load vgg successfully.')
    elif model_type == 'resnet':
        model = resnet18(is_ptrtrained=False)
        print('load resnet18 successfully.')
    else:
        raise ValueError('')
    model.load_state_dict(remove_prefix(checkpoint['state_dict']))
    return model


def preprocess(path):
    """
    images in custom-defiend skin dataset end with suffix .jpg while images in DR ends with suffix .jpeg
    :param path:
    :return:
    """
    if '.jpeg' in path:
        mean = [0.651, 0.4391, 0.2991]
        std = [0.1046, 0.0846, 0.0611]
    elif '.jpg' in path:
        mean = [0.7432, 0.661, 0.6283]
        std = [0.0344, 0.0364, 0.0413]
    else:
        raise ValueError('')
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    img_pil = Image.open(path)
    return transform(img_pil).unsqueeze(0)


def main(data_dir, pretrained_path, model_type, saved_path, suffix):
    model = load_pretrained_model(pretrained_path, model_type)
    model.eval()
    results = classifiy(data_dir, model, suffix)
    print(str(results))
    save_results(results, os.path.join(saved_path, suffix))


def save_results(results, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    write(results, '%s/results.txt' % (saved_path))


def classifiy(data_dir, model, suffix):
    results = dict()
    for phase in ['lesion', 'normal']:
        path = '%s/%s_data_%s' % (data_dir, phase, suffix)
        total_nums = len(os.listdir(path))
        for image_idx, name in enumerate(os.listdir(path), 1):
            abs_path = os.path.join(path, name)
            img_tensor = preprocess(abs_path)
            classes = {'lesion': 0,
                       'normal': 1}
            logit = model(img_tensor)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            label = 0 if 'lesion' in name else 1
            results[name] = dict(label=label, pred=idx[0].item(), prob=probs[0].item())
            if image_idx % 50 == 0:
                print('%s:[%d/%d]' % (phase, image_idx, total_nums))
    return results


if __name__ == '__main__':
    # classify dataset
    main(data_dir='../gan174/all_results_499/', pretrained_path='../classifier06',
         saved_path='../gan174/all_results_499/after_training',
         model_type='resnet', suffix='single')
    # note parameter should be in ['original', 'single']
    main(data_dir='../gan174/all_results_499/', pretrained_path='../classifier06',
         saved_path='../gan174/all_results_499/after_training',
         model_type='resnet', suffix='original')
