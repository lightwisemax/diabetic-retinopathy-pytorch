import os
import sys

from sklearn.metrics import confusion_matrix

sys.path.append('../')
from utils.util import read

def main(dict_path):
    classes = {'lesion': 0,
               'normal': 1}
    data_dir = '../data/target_128'
    pred_dict = read(dict_path)
    results = dict()
    for phase in ['train', 'val']:
        y_true = []
        y_pred = []
        parent_folder = os.path.join(data_dir, phase)
        for name in os.listdir(parent_folder):
            if 'lesion' in name:
                y_true.append(0)
            elif 'normal' in name:
                y_true.append(1)
            else:
                raise ValueError('')
            y_pred.append(pred_dict[name]['pred'])
        cm = confusion_matrix(y_pred, y_true)
        TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        results[phase] = dict(recall=recall, specificity=specificity)
    return results


if __name__ == '__main__':
    for suffix in ['single', 'original']:
        print(suffix, main('../gan174/all_results_499/after_training/%s/results.txt' % suffix))