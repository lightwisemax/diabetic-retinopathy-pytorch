#!/usr/bin/env bash
#after training:此处用的数据:normal:所有的normal数据 lesion:所有的unet输出(原始lesion图片作为输入) normalize用的是原始数据集的mean和std
#注:需要先把unet output保存到local(包括lesion和normal图片) 再在原始数据集上训练一个分类器
#另外：(1)normalize的mean和std改为新数据集是否会有提升
#     (2)是否只在测试集上来计算after training
python after_training.py ../gan174/all_results_499/lesion_data_single ../classifier01 ../gan174/all_results_499/after_training resnet
CUDA_VISIBLE_DEVICES=2,3 python multi_classifier.py -b=128 -e=100 --lr=1e-3 --step_size=50 -p=classifier02 -m=vgg
#classifier02:grad-cam可视化分类器vgg19 准确率0.93 测试集上
#classifier01:cam可视化分类器resnet18
#classifier03:grad-cam可视化分类器vgg19 准确率0.94 测试集上
python grad_cam.py ../classifier02 ../data/target_128 ../grad_cam01
#save unet outputs as local pm custom-defined skin dataset
python save_all_results.py gan268 1099 ../data/contrast_dataset false
#save dice loss on custom-defined skin dataset
#在gan262上计算
python dice_loss.py gan262 499 ../data/contrast_dataset
#在gan268上计算
python dice_loss.py gan268 1099 ../data/contrast_dataset
#在custom-defined skin dataset计算after training
python after_training.py ../gan268/all_results_1099/ ../classifier04 ../gan268/all_results_1099/after_training resnet
#在custom-defined skin dataset上训练resnet18
CUDA_VISIBLE_DEVICES=4,5 python multi_classifier.py -b=128 -e=100 --lr=1e-3 --step_size=50 -p=classifier04 -m=resnet18 -d=./data/split_contrast_dataset
#在原始DR上训练resnet18
CUDA_VISIBLE_DEVICES=5 python multi_classifier.py -b=64 -e=100 --lr=1e-3 --step_size=50 -p=classifier01 -m=resnet18

CUDA_VISIBLE_DEVICES=2 python multi_classifier.py -b=128 -e=100 --lr=1e-3 --step_size=50 -p=classifier05 -m=vgg_cam -d=./data/target_128
#对uNet输出进行分类 看是否出现过拟合情况 结果发现在training数据集上近乎100% 然后validation集上75%
CUDA_VISIBLE_DEVICES=0 python test_over_fitting.py -b=64 -e=100 --lr=1e-3 --step_size=50 -p=classifier09 -m=resnet -d=./contrast/gan174_output
