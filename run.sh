#!/bin/bash
#CUDA_VISIBLE_DEVICES=5,4 python u_d.py -ts=wgan-gp -b=64 -e=390 -a=0.5 -l=1.0 -k=2 --gan_type=multi_scale -p gan145 -u --gamma=5.0 --beta1=0.5 -i=15 -n=1 --pretrained_epochs=1
#CUDA_VISIBLE_DEVICES=0,1,2 python locate.py -b=210 --lr=1e-3 --lmbda=0.9 --step_size=40 --gamma=0.1 --interval_freq=12 -k=2 -p=locator41 -e=160
#CUDA_VISIBLE_DEVICES=1 python u_d_c.py -b=32 -e=499 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=2 -p gan174 -u --beta1=0.3 -i=10 --gan_type=multi_scale --eta=25.0 --pretrained_steps=1950 --epsi=0.996 --theta=100.0

# CUDA_VISIBLE_DEVICES=1,2 python locate.py -b=140 --lr=1e-4 --lmbda=0.9 --step_size=80 --gamma=0.1 --interval_freq=12 -k=2 -p=locator41 -e=160
# CUDA_VISIBLE_DEVICES=4,5,6 python locate.py -b=210 --lr=1e-4 --lmbda=0.96 --step_size=80 --gamma=0.1 --interval_freq=12 -k=2 -p=locator43 -e=160
# locate: mse_loss
# locate1: l1_loss
#CUDA_VISIBLE_DEVICES=6,7 python locate1.py -b=140 --lr=1e-4 --lmbda=0.9 --step_size=200 --gamma=0.1 --interval_freq=12 -k=2 -p=locator53 -e=160
#CUDA_VISIBLE_DEVICES=0 python locate1.py -b=30 --lr=1e-4 --lmbda=0.96 --step_size=200 --gamma=0.1 --interval_freq=12 -k=2 -p=locator51 -e=160
#CUDA_VISIBLE_DEVICES=0 python locate1.py -b=30 --lr=1e-4 --lmbda=0.8 --step_size=200 --gamma=0.1 --interval_freq=12 -k=2 -p=locator52 -e=160

#CUDA_VISIBLE_DEVICES=5 python multi_classifier.py -b=64 -e=100 --lr=1e-3 --step_size=50 -p=classifier01 -m=resnet18
#CUDA_VISIBLE_DEVICES=4,5,6,7 python identical_mapping.py -b=256 -e=100 -i=1 -p=identical_mapping59 -a=0 --lr=1e-4 -d=./data/split_contrast_dataset

CUDA_VISIBLE_DEVICES=4,5,6,7 python u_d_c.py -b=128 -e=2 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=0 -p test -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=25.0 --pretrained_steps=0 --epsi=0.996 --theta=10.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset --debug

CUDA_VISIBLE_DEVICES=4,5,6,7 python u_d_c.py -b=128 -e=499 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=0 -p gan262 -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=1.0 --pretrained_steps=0 --epsi=0.996 --theta=10.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset
#CUDA_VISIBLE_DEVICES=4,5,6,7 python u_d_c.py -b=128 -e=499 -a=1.0 -g=1.0 -l=5.0 -s=0.2 -k=0 -p gan263 -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=10.0 --pretrained_steps=0 --epsi=0.996 --theta=10.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset
#CUDA_VISIBLE_DEVICES=4,5,6,7 python u_d_c.py -b=128 -e=499 -a=1.0 -g=1.0 -l=1.0 -s=1.0 -k=0 -p gan264 -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=0.0 --pretrained_steps=3 --epsi=0.996 --theta=10.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset
#CUDA_VISIBLE_DEVICES=0,1,2,3 python u_d_c.py -b=128 -e=499 -a=1.0 -g=5.0 -l=10.0 -s=1.0 -k=0 -p gan265 -u --beta1=0.5 -i=1 --gan_type=multi_scale --eta=0.0 --pretrained_steps=3 --epsi=0.996 --theta=10.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset

#CUDA_VISIBLE_DEVICES=1,2,3 python u_d_c.py -b=96 -e=1099 -a=1.0 -g=15.0 -l=10.0 -s=0.4 -k=0 -p gan266 -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=1.0 --pretrained_steps=15 --epsi=0.996 --theta=25.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset
#gan262 gan266 效果还行  gan263 gan264 gan265效果较差
#CUDA_VISIBLE_DEVICES=4,5,6,7 python u_d_c.py -b=128 -e=1099 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=2 -p gan267 -u --beta1=0.3 -i=10 --gan_type=multi_scale --eta=25.0 --pretrained_steps=100 --epsi=0.996 --theta=100.0 --data=./data/contrast_dataset

#CUDA_VISIBLE_DEVICES=2,3 python multi_classifier.py -b=128 -e=100 --lr=1e-3 --step_size=50 -p=classifier02 -m=vgg

#CUDA_VISIBLE_DEVICES=0,1,2,3 python u_d_c.py -b=128 -e=1099 -a=1.0 -g=15.0 -l=10.0 -s=0.4 -k=0 -p gan268 -u --beta1=0.3 -i=1 --gan_type=multi_scale --eta=1.0 --pretrained_steps=650 --epsi=0.996 --theta=25.0 --pretrain_unet_path=./identical_mapping59/identical_mapping.pkl --data=./data/contrast_dataset
#最后实验结果发现 gan268效果最佳


#对比没有total variation loss情况
CUDA_VISIBLE_DEVICES=6 python u_d_c.py -b=32 -e=499 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=2 -p gan290 -u --beta1=0.3 -i=10 --gan_type=multi_scale --eta=25.0 --pretrained_steps=1950 --epsi=0.996 --theta=0.0 --pretrain_unet_path=./identical_mapping45/identical_mapping.pkl --local --data=/data/zhangrong/gan
CUDA_VISIBLE_DEVICES=1 python u_d_c.py -b=32 -e=499 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=2 -p gan291 -u --beta1=0.3 -i=10 --gan_type=multi_scale --eta=25.0 --pretrained_steps=1950 --epsi=0.996 --theta=0.0 --pretrain_unet_path=./identical_mapping45/identical_mapping.pkl --local --data=/data/zhangrong/gan -ts=add_normal_constraint
CUDA_VISIBLE_DEVICES=2 python u_d_c.py -b=32 -e=499 -a=1.0 -g=10.0 -l=10.0 -s=0.4 -k=2 -p gan292 -u --beta1=0.3 -i=10 --gan_type=multi_scale --eta=25.0 --pretrained_steps=1950 --epsi=0.996 --theta=0.0 --pretrain_unet_path=./identical_mapping45/identical_mapping.pkl --local --data=/data/zhangrong/gan -ts=remove_lesion_constraint

#对比只用normal数据来训练UNet 看能否locate biomarker
CUDA_VISIBLE_DEVICES=3,4 python AE_detect.py -b=128 -e=150 -i=1 -p=ae_detect01 -a=0 --lr=1e-4 --local