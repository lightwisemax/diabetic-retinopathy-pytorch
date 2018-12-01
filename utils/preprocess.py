import cv2
import os
import numpy as np


def adjust_contrast(img):
    # Convert to signed 16 bit. this will allow values less than zero and
    # greater than 255
    img = np.int16(img)

    contrast = 64
    brightness = 0

    img = img * (contrast / 127 + 1) - contrast + brightness

    # we now have an image that has been adjusted for brightness and
    # contrast, but we need to clip values not in the range 0 to 255
    img = np.clip(img, 0, 255)  # force all values to be between 0 and 255
    # finally, convert image back to unsigned 8 bit integer
    img = np.uint8(img)


def adjust_gamma(image, gamma=1.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def test_gamma_correction():
    # Open a typical 24 bit color image. For this kind of image there are
    # 8 bits (0 to 255) per color channel
    data_dir = '../data/diabetic_test/training'
    for name in os.listdir(data_dir):
        abs_path=os.path.join(data_dir, name)
        img = cv2.imread(abs_path)  # mandrill reference image from USC SIPI
        img = adjust_gamma(img, gamma=2.0)
        cv2.imwrite(abs_path, img)


if __name__ == '__main__':
    test_gamma_correction()
