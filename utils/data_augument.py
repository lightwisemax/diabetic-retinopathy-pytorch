"""
use ImageEnhance model to da data augument
"""
import os
import cv2
import PIL
import random
from PIL import ImageEnhance, Image

def color(im, factor):
    assert isinstance(im, PIL.JpegImagePlugin.JpegImageFile), ''
    return ImageEnhance.Color(im).enhance(factor)

def brightness(im, factor):
    assert isinstance(im, PIL.JpegImagePlugin.JpegImageFile), ''
    return ImageEnhance.Brightness(im).enhance(factor)

def horizontal_flip(path, saved_path):
    im = cv2.imread(path)
    h_filp = cv2.flip(im, 1)
    cv2.imwrite(saved_path, h_filp)

def data_augument():
    source_path = '../data/gan/normal'
    source_lst = os.listdir(source_path)[:16]
    for name in source_lst:
        abs_path = os.path.join(source_path, name)
        im = Image.open(abs_path)
        im = brightness(im, random.random() + 0.5)
        # print(os.path.join(source_path, '%s_birghtness.%s' % (name.split('.')[0], name.split('.')[1])))
        im.save(os.path.join(source_path, '%s_birghtness.%s' % (name.split('.')[0], name.split('.')[1])))

def flip(source_path, target_path):
    """
    flip images on the left side of the disc
    :return:
    """
    for name in os.listdir(source_path):
        if '.jpeg' in name:
            abs_path = os.path.join(source_path, name)
            saved_path = os.path.join(target_path, name)
            horizontal_flip(abs_path, saved_path)
            print('save %s to %s successfully.'  %(name, saved_path))


if __name__ == '__main__':
    # flip(source_path = '../data/gan_h_flip/flip_lesion/right', target_path = '../data/gan_h_flip/flip_lesion/h_flip')
    # flip(source_path = '../data/gan_h_flip/flip_normal/right', target_path = '../data/gan_h_flip/flip_normal/h_flip')
    lis = ['normal_3472_left.jpeg', 'normal_6836_right.jpeg', 'normal_7270_right.jpeg', 'normal_7359_right.jpeg', 'normal_7558_right.jpeg', 'normal_8496_left.jpeg', 'normal_13564_left.jpeg', 'normal_14026_left.jpeg', 'normal_16205_right.jpeg', 'normal_20232_right.jpeg', 'normal_23456_right.jpeg', 'normal_24665_left.jpeg', 'normal_24968_right.jpeg', 'normal_25005_right.jpeg', 'normal_25148_right.jpeg', 'normal_25832_left.jpeg', 'normal_25992_left.jpeg', 'normal_26894_left.jpeg', 'normal_27112_right.jpeg', 'normal_27458_right.jpeg', 'normal_27912_left.jpeg', 'normal_27817_right.jpeg', 'normal_29479_right.jpeg', 'normal_33156_right.jpeg', 'normal_39000_right.jpeg', 'normal_38949_right.jpeg', 'normal_39022_left_birghtness.jpeg', 'normal_39417_right.jpeg', 'normal_41589_right.jpeg']
    for name in lis:
        abs_path = os.path.join('../data/gan_h_flip/normal', name)
        im = cv2.imread(abs_path)
        h_filp = cv2.flip(im, 1)

        cv2.imwrite(abs_path, h_filp)