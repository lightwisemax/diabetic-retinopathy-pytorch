"""
compute confusion matrix of training set when training classifier on DR.However,this procession should be done during training rather than an additional script.
"""
import os
import sys

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('../')
from contrast.roc_curve import load_pretrained_model
from utils.read_data import EasyDR


def main(prefix, data_dir):
    model = load_pretrained_model(prefix, 'resnet')
    model.eval()
    train_loader, val_loader = load_dataset(data_dir)

    val_recall, val_specificity = compute(model, val_loader)
    print(val_recall, val_specificity)

    train_recall, train_specificity = compute(model, train_loader)
    print(train_recall, train_specificity)
    print('\trecall\tspecificity')
    print('train\t%.4f\t%.4f' %(train_recall, train_specificity))
    print('val\t%.4f\t%.4f' %(val_recall, val_specificity))


def compute(model, loader):
    pred_y = []
    test_y = []
    for inputs, labels, _, _ in (loader):
        outputs = model(inputs)
        pred_y.extend(outputs.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
        test_y.extend(labels.data.cpu().numpy().tolist())
    # print('there are %d images totally.' % len(pred_y))
    cm = confusion_matrix(pred_y, test_y)
    TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return TP / (TP + FN), TN / (TN + FP)


def load_dataset(data_dir):
    if data_dir == '../data/target_128':
        mean = [0.651, 0.4391, 0.2991]
        std = [0.1046, 0.0846, 0.0611]
        print('load DR with 128 successfully!!!')
    else:
        raise ValueError("parameter 'data' that means path to dataset must be in "
                         "['./data/target_128', ./data/split_contrast_dataset]")
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean, std)
    post_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = EasyDR(traindir, None, post_transforms, alpha=0)
    val_dataset = EasyDR(valdir, None, post_transforms, alpha=0)

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=False)
    return train_loader, val_loader


if __name__ == '__main__':
    main(prefix='../classifier06', data_dir = '../data/target_128')
