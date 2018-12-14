import os
import shutil

source_train = os.listdir('data/target_128/train')
source_val = os.listdir('data/target_128/val')

target_path = 'contrast/gan174_output'
target_lesion_path = os.path.join(target_path, 'lesion_data_single')
target_lesion = os.listdir(target_lesion_path)

target_normal_path = os.path.join(target_path, 'normal_data_single')
target_normal = os.listdir(target_normal_path)

for name in target_lesion:
    if '.jpeg' not in name:
        continue
    if name in source_train:
        shutil.copy(os.path.join(target_lesion_path, name), os.path.join(target_path, 'train'))
    elif name in source_val:
        shutil.copy(os.path.join(target_lesion_path, name), os.path.join(target_path, 'val'))
    else:
        raise ValueError('')

for name in target_normal:
    if '.jpeg' not in name:
        continue
    if name in source_train:
        shutil.copy(os.path.join(target_normal_path, name), os.path.join(target_path, 'train'))
    elif name in source_val:
        shutil.copy(os.path.join(target_normal_path, name), os.path.join(target_path, 'val'))
    else:
        print(name)
        # raise ValueError('')

