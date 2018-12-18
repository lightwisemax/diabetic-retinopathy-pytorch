import os
import cv2
from PIL import Image

from dataset.diabetic_retinopathy.split_ds import Split
from utils.util import clear


class CenterCrop(object):
    """
    this script will preprocess the dataset with opencv in advance and save to disk
    rather than do preprocession with dataloader in such case that
    the speed of data is too slow(e.g.:it takes 130 seconds to load 130 images).
    Example:
        im = CenterCrop()('./processed/LESION/15_left.jpeg', 3000, 3000)
        im.save('test.png')
    """
    def __init__(self):
        pass

    def __call__(self, path, new_height, new_width):
        image = Image
        width, height = image.size
        if new_height > height or new_width > width:
            raise RuntimeError('target size is smaller than source image size!!!')
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return image.crop((left, top, right, bottom))


class AspectRatio(object):
    """
    Take the max width and length for given images then put the image in that size to resize effective
    reference: https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape/43391469
               https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    note: this result will overwrite original image
    Examples:
        AspectRatio()('./processed/LESION_REC')
    """
    def __init__(self):
        pass

    def __call__(self, source):
        for name in os.listdir(source):
            abs_path = os.path.join(source, name)
            self.aspect_ratio(abs_path)

    @staticmethod
    def aspect_ratio(path):
        img = cv2.imread(path)
        # old_size is in (height, width) format
        old_size = img.shape[:2]
        # this operation just is applied on image with differe height and width
        if old_size[0] is not old_size[1]:
            desired_size = max(old_size)
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            # new_size should be in (width, height) format
            img = cv2.resize(img, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            # add zeros around the image
            new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            print('save to %s' % path)
            cv2.imwrite(path, new_im)


class ExternalRectangle(object):
    """
    obtain diabetic retinopathy external rectangle to cut valueless information(i.e. background)
    Examples:
        ExternalRectangle()(root='./processed/LESION', target_root='./processed/LESION_REC')
    """
    def __init__(self):
        pass

    def __call__(self, source, target):
        if os.path.exists(target):
            clear(target)
        else:
            os.makedirs(target)
        for name in os.listdir(source):
            self.crop(target, os.path.join(source, name))

    @staticmethod
    def crop(root, path, thresh=25, offset=0):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        """
            obtain binary mask
            @:param src: gray image
            @:param thresh: thresh value
            @:param type: binary value
            @:return ret: thresh
            @:return thresh: binary mask
        """
        ret, thresh = cv2.threshold(src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
        # compute external rectangle and return size(i.e. height and width) and position(i.e. x and y)
        x, y, w, h = cv2.boundingRect(thresh)
        bounding = image[y:(y + h), (x + offset):(x + w - offset), :]
        saved_path = '%s/rec_%s' % (root, path.split('/')[-1])
        print('save to %s' % saved_path)
        cv2.imwrite(saved_path, bounding)


class Resize(object):
    def __init__(self):
        pass

    def __call__(self, path, new_size):
        path_lst = os.listdir(path)
        for name in path_lst:
            abs_path = os.path.join(path, name)
            img = cv2.imread(abs_path)
            assert img.shape[0] == img.shape[1]
            img = cv2.resize(img, new_size)
            cv2.imwrite(abs_path, img)
            print('save to %s' % abs_path)


class MaxInscribedRectangle(object):
    """
        compute maximum inscribed rectangle based on minimum inscribed rectangle
        call for NORMAL and LESION image after adjusting aspect ratio and before spliting dataset
        the operation step is as follows if use max inscribed rectangle:
        1.extract alternatively normal and lesion images for below training
        2.obtain max inscribed rectangle for every image
        3.split dataset
        the three steps above is implemented in one class called EliminateBoundary in order to make dataset conveniently
    """
    def __init__(self):
        pass

    def __call__(self, source_path, saved_path, thresh=25):
        """
        :param source_path: images source path
        :param saved_path: final result saved path
        :param thresh:
        :return:
        """
        if os.path.exists(saved_path):
            clear(source_path)
        else:
            os.makedirs(saved_path)
        lst = os.listdir(source_path)
        for name in lst:
            abs_path = os.path.join(source_path, name)
            self.compute_inscribed_rec(abs_path, saved_path, thresh)

    @staticmethod
    def compute_inscribed_rec(path, saved_path, thresh):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
        # compute external rectangle and return size(i.e. height and width) and position(i.e. x and y)
        x, y, w, h = cv2.boundingRect(thresh)
        pos_x = x + int(w / 2)
        pos_y = y + int(h / 2)
        r = (w / 2 + h / 2) / 2
        side_len = int(r / 1.414)
        inscribed_rec = image[pos_y - side_len: pos_y + side_len, pos_x - side_len:pos_x + side_len, :]
        saved_path = '%s/in_%s' % (saved_path, path.split('/')[-1])
        print('save to %s' % saved_path)
        cv2.imwrite(saved_path, inscribed_rec)


class EliminateBoundary(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        # MaxInscribedRectangle()(source_path='./processed/NORMAL', saved_path='./processed/NORMAL_WB')
        # MaxInscribedRectangle()(source_path='./processed/LESION', saved_path='./processed/LESION_WB')
        # TODO:need to copy extra image to target folder manually
        # self.extra()
        Split({'NORMAL': './processed/NORMAL_WB', 'LESION': './processed/LESION_WB'},
              './processed/train_wb',
              './processed/val_wb',
              new_size=(256, 256),
              is_resize=True)()

    def inscribed_rec(path, thresh, phase):
        print(path)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
        # print(thresh)
        # compute external rectangle and return size(i.e. height and width) and position(i.e. x and y)
        x, y, w, h = cv2.boundingRect(thresh)
        pos_x = x + int(w / 2)
        pos_y = y + int(h / 2)
        r = (w / 2 + h / 2) / 2
        side_len = int(r / 1.414)
        in_rec = image[pos_y - side_len: pos_y + side_len, pos_x - side_len:pos_x + side_len, :]
        cv2.imwrite('%s/in_%s' % (phase, path.split('/')[-1]), in_rec)

    def extra(self):
        normal_lst = ['in_280_left.jpeg', 'in_356_right.jpeg', 'in_636_right.jpeg', 'in_921_left.jpeg',
                      'in_979_left.jpeg', 'in_2721_right.jpeg',
                      'in_2886_left.jpeg', 'in_3232_left.jpeg', 'in_3862_right.jpeg', 'in_3909_right.jpeg',
                      'in_4631_left.jpeg', 'in_4734_left.jpeg',
                      'in_5459_left.jpeg', 'in_5668_left.jpeg', 'in_5991_right.jpeg', 'in_6423_right.jpeg',
                      'in_7022_right.jpeg', 'in_7722_left.jpeg', 'in_7962_left.jpeg',
                      'in_8153_left.jpeg', 'in_8754_left.jpeg', 'in_9099_left.jpeg', 'in_9148_right.jpeg',
                      'in_9739_right.jpeg',
                      'in_9996_right.jpeg', 'in_10162_right.jpeg', 'in_10367_right.jpeg', 'in_10407_left.jpeg',
                      'in_10548_right.jpeg', 'in_11301_left.jpeg',
                      'in_11301_right.jpeg', 'in_13685_right.jpeg', 'in_13770_right.jpeg', 'in_13808_left.jpeg',
                      'in_14063_left.jpeg',
                      'in_14610_right.jpeg', 'in_15516_left.jpeg', 'in_15949_right.jpeg', 'in_17746_left.jpeg',
                      'in_17942_right.jpeg', 'in_18984_right.jpeg',
                      'in_19641_right.jpeg', 'in_19888_right.jpeg', 'in_20043_left.jpeg', 'in_21876_right.jpeg',
                      'in_21294_left.jpeg', 'in_20782_left.jpeg', 'in_20753_left.jpeg',
                      'in_22267_left.jpeg', 'in_22287_right.jpeg', 'in_22452_right.jpeg', 'in_22857_left.jpeg',
                      'in_23560_right.jpeg', 'in_23597_right.jpeg',
                      'in_25261_right.jpeg', 'in_25395_right.jpeg', 'in_25781_right.jpeg', 'in_26801_right.jpeg',
                      'in_27659_left.jpeg', 'in_28019_left.jpeg',
                      'in_28033_left.jpeg', 'in_28918_right.jpeg', 'in_29530_left.jpeg', 'in_29579_left.jpeg',
                      'in_30306_left.jpeg', 'in_30904_right.jpeg',
                      'in_35792_left.jpeg', 'in_35712_left.jpeg', 'in_35325_right.jpeg', 'in_35005_right.jpeg',
                      'in_34872_right.jpeg', 'in_33394_right.jpeg', 'in_33081_right.jpeg',
                      'in_35836_left.jpeg', 'in_35898_left.jpeg', 'in_36409_left.jpeg', 'in_36411_left.jpeg',
                      'in_38014_right.jpeg', 'in_38213_left.jpeg', 'in_38349_left.jpeg',
                      'in_41695_left.jpeg', 'in_42117_right.jpeg', 'in_42393_left.jpeg', 'in_42393_right.jpeg',
                      'in_43467_left.jpeg', 'in_38866_left.jpeg', 'in_38945_right.jpeg', 'in_39426_right.jpeg',
                      'in_39367_right.jpeg', 'in_39367_left.jpeg', 'in_39988_left.jpeg', 'in_39988_left.jpeg',
                      'in_40285_left.jpeg',
                      'in_43470_right.jpeg', 'in_43758_right.jpeg', 'in_44185_right.jpeg']
        lesion_lst = ['in_16007_right.jpeg', 'in_13387_left.jpeg', 'in_13387_right.jpeg', 'in_11730_right.jpeg',
                      'in_10321_left.jpeg',
                      'in_16309_right.jpeg', 'in_16309_left.jpeg', 'in_18803_right.jpeg', 'in_19285_right.jpeg',
                      'in_24346_left.jpeg', 'in_30722_left.jpeg', 'in_30722_right.jpeg',
                      'in_32851_right.jpeg', 'in_32507_left.jpeg', 'in_31712_right.jpeg', 'in_31202_right.jpeg',
                      'in_31202_left.jpeg',
                      'in_38856_left.jpeg', 'in_38407_right.jpeg', 'in_38407_left.jpeg', 'in_37951_left.jpeg',
                      'in_37951_right.jpeg',
                      'in_38856_left.jpeg', 'in_38856_right.jpeg', 'in_40917_left.jpeg', 'in_42815_left.jpeg',
                      'in_42815_right.jpeg', 'in_43998_left.jpeg']
        phase = 'normal'
        for name in normal_lst:
            name = name[3:]
            abs_path = os.path.join('./dataset/diabetic_retinopathy/processed/NORMAL', name)
            self.inscribed_rec(abs_path, 110, phase)
        phase = 'lesion'
        for name in lesion_lst:
            name = name[3:]
            abs_path = os.path.join('./dataset/diabetic_retinopathy/processed/LESION', name)
            self.inscribed_rec(abs_path, 110, phase)




if __name__ == '__main__':
    """
        aspect ratio then crop external rectangle
        crop external rectangle then aspect ratio
    """
    # im = CenterCrop()('./processed/LESION/15_left.jpeg', 3000, 3000)
    # im.save('test.jpeg')

    # ExternalRectangle()('./processed/LESION', './processed/LESION_REC')
    # ExternalRectangle()('./processed/NORMAL', './processed/NORMAL_REC')
    # AspectRatio()('./processed/LESION_REC')
    # AspectRatio()('./processed/NORMAL_REC')
    # EliminateBoundary()()
    pass
