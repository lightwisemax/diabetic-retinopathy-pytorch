import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class Point(object):
    def __init__(self , x , y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y

class SRG(object):
    def __init__(self, path, saved_path):
        self.path = path
        self.im = cv2.imread(path)
        self.saved_path = saved_path

    def get_dist(self, seed_location1, seed_location2):
        l1 = self.im[seed_location1.x, seed_location1.y]
        l2 = self.im[seed_location2.x, seed_location2.y]
        count = np.sqrt(np.sum(np.square(l1 - l2)))
        return count

    def __call__(self):
        im_shape = self.im.shape
        height = im_shape[0]
        width = im_shape[1]
        connects = [ Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
        img_mark = np.zeros([height , width])

        # 建立空的图像数组,作为一类
        img_re = self.im.copy()
        for i in range(height):
            for j in range(width):
                img_re[i, j][0] = 0
                img_re[i, j][1] = 0
                img_re[i, j][2] = 0
        #随即取一点作为种子点
        seed_list = []
        seed_list.append(Point(10, 10))
        T = 7#阈值
        class_k = 1#类别
        #生长一个类
        while (len(seed_list) > 0):
            seed_tmp = seed_list[0]

            seed_list.pop(0)
            img_mark[seed_tmp.x, seed_tmp.y] = class_k

            # 遍历8邻域
            for i in range(8):
                tmpX = seed_tmp.x + connects[i].x
                tmpY = seed_tmp.y + connects[i].y

                if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                    continue
                dist = self.get_dist(seed_tmp, Point(tmpX, tmpY))
                #在种子集合中满足条件的点进行生长
                if (dist < T and img_mark[tmpX, tmpY] == 0):
                    img_re[tmpX, tmpY][0] = self.im[tmpX, tmpY][0]
                    img_re[tmpX, tmpY][1] = self.im[tmpX, tmpY][1]
                    img_re[tmpX, tmpY][2] = self.im[tmpX, tmpY][2]
                    img_mark[tmpX, tmpY] = class_k
                    seed_list.append(Point(tmpX, tmpY))

        self._save(self.saved_path, img_re)

    def _save(self, saved_path, img_re):

        # img_re = cv2.cvtColor(img_re, cv2.COLOR_RGB2GRAY)
        # img_re = np.where(img_re==0, 255, 0).astype(np.uint8)
        # kernel = np.ones((2, 2),np.uint8)
        # optic_disk = cv2.erode(img_re,kernel,iterations = 1)
        # optic_disk = np.where(optic_disk==255, 0, 1)
        # cv2.imwrite(saved_path , img_re * optic_disk)
        cv2.imwrite(saved_path, img_re)



if __name__ == '__main__':
    root = '../data/gan_h_flip/lesion'
    target = '../data/gan_h_flip/mask'
    for name in os.listdir(root):
        path = os.path.join(root, name)
        print(path)
        saved_path = os.path.join(target, name)
        print(saved_path)
        SRG(path, saved_path)()

