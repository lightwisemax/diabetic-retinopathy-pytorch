"""
reference: https://github.com/znxlwm/pytorch-generative-model-collections
"""
import argparse
import sys

sys.path.append('./')
from training_strategies import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training Custom Defined Model')
    parser.add_argument('--training_strategies', '-ts', default='update_c_d_u', choices=['update_c_d_u'], help='training strategies')
    parser.add_argument('--debug', action='store_true', default=False, help='mode:training or debug')
    parser.add_argument('--prefix', '-p', type=str, required=True, help='parent folder to save output data')
    parser.add_argument('--is_pretrained_unet', '-u', action='store_true', help='pretrained unet or not')
    parser.add_argument('--pretrain_unet_path', type=str, default='./identical_mapping45/identical_mapping.pkl', help='pretrained unet')
    parser.add_argument('--power', '-k', type=int, default=2, help='power of weight')
    parser.add_argument('--data', type=str, default='./data/gan', choices=['./data/gan'], help='dataset type')
    parser.add_argument('--batch_size', '-b', default=64, type=int, required=True, help='batch size')
    parser.add_argument('--gan_type', type=str, default='conv_bn_leaky_relu',
                        choices=['conv_bn_leaky_relu', 'resnet', 'multi_scale'],
                        help='discriminator type')
    parser.add_argument('--u_depth', type=int, default=5, help='unet dpeth')
    parser.add_argument('--d_depth', type=int, default=7, help='discriminator depth')
    parser.add_argument('--dowmsampling', type=int, default=4, help='dowmsampling times in discriminator')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 in Adam')
    parser.add_argument('--interval', '-i', default=15, type=int, required=True, help='log print interval')
    parser.add_argument('--epochs', '-e', default=390, type=int, required=True, help='training epochs')
    parser.add_argument('--lmbda', '-l', type=float, help='weight of u between u and c')
    parser.add_argument('--alpha', '-a', type=float, help='weight of d in c & u & d')
    parser.add_argument('--gamma', '-g', type=float, help='weight of u in c & u & d')
    parser.add_argument('--eta', type=float, default=10.0, help='gradient penalty')
    parser.add_argument('--pretrained_steps', type=int, default=0, help='pretrained steps')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.training_strategies == 'update_c_d_u':
        trainer = update_c_d_u(args)
        script_path = './training_strategies/update_c_d_u.py'
        print('training step:')
        print('(1)fix D, update C & U')
        print('2)fix C & G, update D')
        print('3)fix C & D, update U')
    else:
        raise ValueError("the training strategies must in ['update_u_c_d_u']")

    trainer.save_running_script(script_path)
    trainer.main()
    trainer.save_log()


if __name__ == '__main__':
    main()