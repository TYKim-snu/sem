import sys
import torch

import argparse
from train import *


class PythonPredictor:

    # def __init__(self, config):
    #
    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

        parser.add_argument('--backbone', type=str, default='resnet',
                            choices=['resnet', 'xception', 'drn', 'mobilenet'],
                            help='backbone name (default: resnet)')
        parser.add_argument('--out-stride', type=int, default=16,
                            help='network output stride (default: 8)')
        parser.add_argument('--dataset', type=str, default='samsung_SEM',
                            choices=['pascal', 'coco', 'cityscapes', 'samsung_SEM', 'samsung_SEM_crop_256'],
                            help='dataset name (default: pascal)')
        parser.add_argument('--use-sbd', action='store_true', default=False,
                            help='whether to use SBD dataset (default: False)')
        parser.add_argument('--workers', type=int, default=0,

                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=1024,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=300,
                            help='crop image size')
        parser.add_argument('--sync-bn', type=bool, default=None,
                            help='whether to use sync bn (default: auto)')
        parser.add_argument('--freeze-bn', type=bool, default=False,
                            help='whether to freeze bn parameters (default: False)')
        parser.add_argument('--loss-type', type=str, default='ce',
                            choices=['ce', 'focal'],
                            help='loss func type (default: ce)')
        # training hyper params
        parser.add_argument('--epochs', type=int, default=500, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                                    training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=32,
                            metavar='N', help='input batch size for \
                                    testing (default: auto)')
        parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                            help='whether to use balanced weights (default: False)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            choices=['poly', 'step', 'cos'],
                            help='lr scheduler mode: (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
                            metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--nesterov', action='store_true', default=False,
                            help='whether use nesterov (default: False)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
        False, help='disables CUDA training')
        parser.add_argument('--gpu-ids', type=str, default='0',
                            help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default=None,
                            help='set the checkpoint name')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval-interval', type=int, default=1,
                            help='evaluuation interval (default: 1)')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')

        args = parser.parse_args()
        args.backbone_pretrained = None
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            try:
                args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        args.useCPU = False
        args.batch_size = 2
        args.lr = 0.001
        # args.base_size = 400
        # args.crop_size = 400
        args.epochs = 10
        args.seed = 999
        args.out_stride = 16
        args.backbone = 'resnet101_CBAM'
        args.GroupNorm = False
        args.loss_type = 'mse'

        args.checkname = 'resnet101_CBAM_1024_new_model_margin'
        args.save_folder = './weight'
        args.optimizer = 'Adam'
        args.gpu_ids = 0
        args.ft = True

        args.no_val = False
        args.eval_interval = 2

        print(args)

        torch.manual_seed(args.seed)

        args.dataset = './data/1_input'
        args.resume = './weight/iccad/model_best.pth.tar'
        args.Save_dir = './data/result'

        trainer = Trainer(args)
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)
        time.sleep(1)
        trainer.validation(0, use_data='test', save_dir=args.Save_dir)
        trainer.writer.close()

# if __name__ == "__main__":
#     a = PythonPredictor()
#     a.predict()