import os
import cv2

def rename(root):
    name_lst = os.listdir(root)
    for name in name_lst:
        abs_path = os.path.join(root, name)
        if '.jpeg' in name:
            os.rename(abs_path, os.path.join(root, name.replace('lesion',  'distinct_lesion')))

if __name__ == '__main__':
    rename('../data/distinct_lesion')