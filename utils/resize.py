import cv2
import os

def resize(root):
    name_lst = os.listdir(root)
    for name in name_lst:
        abs_path = os.path.join(root, name)
        if '.jpg' in name:
            img = cv2.imread(abs_path)
            cv2.imwrite(abs_path, cv2.resize(img, (128, 128)))
            

if __name__ == '__main__':
    resize('../data/gan11/lesion')
    resize('../data/gan11/normal')