import argparse
import os
import numpy as np
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, Evaluator_for_BE, Evaluator_class

import torch
from torchvision.utils import save_image

import os
import time

class Trainer(object):
    def __init__(self, args):
        self.args = args 

        if args.useCPU is True:
            self.device = torch.device('cpu')
            print('Using device:',  self.device)
        else:
            self.device = torch.device('cuda:'+str(args.gpu_ids))
              # setting device on GPU if available, else CPU
            print('Using device:',  self.device)
            if self.device.type == 'cuda':
                print('GPU CARD {}'.format(str(args.gpu_ids)))
                print(torch.cuda.get_device_name(args.gpu_ids))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(args.gpu_ids)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(args.gpu_ids)/1024**3,1), 'GB')

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(args=args,
                        num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        freeze_bn=args.freeze_bn,
                        GroupNorm=args.GroupNorm)

        self.scheduler = None
        if args.optimizer  is 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer is 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        
        #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

        # Define Criterion
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator

        self.evaluator = Evaluator_for_BE(2)
        self.evaluator_class = Evaluator_class(2)



        #tensorboard add graph it's very heavy !!!!
        #sample = next(iter(self.train_loader))
        #temp_images = sample['image']
        #self.writer.add_graph(self.model, temp_images)

        # Using cuda
        if args.cuda:
            #self.model = self.model.cuda()
            self.model = self.model.to(device=self.device)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location=self.device)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            #print(checkpoint['best_pred'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('reset best pred..')
            self.best_pred = 0.0
        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        if self.scheduler is not None:    
            # Print Learning Rate
            print('Epoch:', epoch,'LR:', self.scheduler.get_lr())
        
        for i, sample in enumerate(tbar):
            image, target ,insert_CAD = sample['image'], sample['label'], sample['insert_CAD']
            image_name = sample['image_name']
            if self.args.cuda:
                #image, target = image.cuda(), target.cuda()
                image, target,insert_CAD = image.to(device=self.device), target.to(device=self.device), insert_CAD.to(device = self.device)
            self.optimizer.zero_grad()
            output = self.model(input =image, input_CAD = insert_CAD)

            # if self.args.dataset.startswith('samsung_SEM_BE') or self.args.dataset.startswith('samsung_CAD_BE'):
            output = torch.sigmoid(output)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show results each epoch
#             if i % (num_img_tr // 10) == 0:
#                 global_step = i + num_img_tr * epoch
#                 self.summary.visualize_image(self.writer, self.args.dataset, 
#                 image.detach().cpu(), target.detach().cpu(), output.detach().cpu(), image_name, global_step)
        
        if self.scheduler is not None:
            # Decay Learning Rate
            self.scheduler.step()

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                #'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch, use_data='val', save_dir = None):
        self.model.eval()
        #self.model.train()
        #self.model.freeze_dropout()
        self.evaluator.reset()

        val_image_save_path = os.path.join(self.summary.directory,'Validation_result')
        if not(os.path.isdir(val_image_save_path)):
            os.makedirs(os.path.join(val_image_save_path))
        with open(os.path.join(self.summary.directory,'Validataion_Missing_Image_List.txt'), 'w') as ff:
            ff.write('')
        
        if use_data is 'val':
            d_loader = self.val_loader
        elif use_data is 'test':
            d_loader = self.test_loader

        tbar = tqdm(d_loader, desc='\r')
        test_loss = 0.0
        
        for i, sample in enumerate(tbar):
            image, target, insert_CAD = sample['image'], sample['label'],sample['insert_CAD']
            image_name = sample['image_name']
            if self.args.cuda:
                #image, target = image.cuda(), target.cuda()
                image, target, insert_CAD = image.to(device=self.device), target.to(device=self.device), insert_CAD.to(device=self.device)
            with torch.no_grad():
                output = self.model(image, insert_CAD)
                # if self.args.dataset.startswith('samsung_SEM_BE') or self.args.dataset.startswith('samsung_CAD_BE'):
                output = torch.sigmoid(output)
            loss = self.criterion(output, target)

            if save_dir != None:
                for i in range(len(output)):
                    save_image(output[i], os.path.join(save_dir,str(image_name[i])))

            test_loss += loss.item()
            tbar.set_description('Validation loss: %.3f' % (test_loss / (i + 1)))

#             self.summary.visualize_image(self.writer, self.args.dataset, 
#                                             image.detach().cpu(), 
#                                             target.detach().cpu(), 
#                                             output.detach().cpu(), 
#                                             image_name,
#                                             i, validation_save=True, show_input_image=False,
#                                             val_ref_image_path=val_ref_image_path,
#                                             task='validation')
            
            pred = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            #print(output.shape)
            
            #save_image(output, os.path.join('/data2/2_ML_team/ml_ty/4_CAD_based_Hotspot_detection/4_valfig'+i+'.tif')
            
            # if self.args.dataset.startswith('samsung_SEM_BE') or self.args.dataset.startswith('samsung_CAD_BE'):
            #     pass
            # else:
            #     pred = np.argmax(pred, axis=1)

            
            #Add batch sample into evaluator
            #print(target.shape)
            if target.shape[0] >= 1:
                #pred_class = np.squeeze(np.reshape(pred.copy(), [pred.shape[0], 1, -1]))
                #target_class = np.squeeze(np.reshape(target.copy(), [target.shape[0], 1, -1]))
                #pred_class = pred_class > 0.3
                #target_class = target_class > 0.3
                #pred_class =  np.sum(pred_class, axis=1)
                #target_class =  np.sum(target_class, axis=1)
                #pred_class = np.array([ 1 if i>0 else 0 for i in pred_class])
                #target_class = np.array([ 1 if i>0 else 0 for i in target_class])
                #print(pred_class)
                #print(target_class)

                self.evaluator.add_batch(target, pred)
                #self.evaluator.add_batch_Clusters_Accurary(target, pred, image_name, self.summary.directory)
                self.evaluator.add_batch_Cad_prediction_Accurary(target, pred, image_name)
                #self.evaluator_class.add_batch(target_class, pred_class)         
            #break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        pixel_acc = self.evaluator.Pixel_Accuracy()
        #Clus_Acc_explict, Clus_Acc_implict = self.evaluator.Clusters_Accurary()
        Cad_predict_Acc = self.evaluator.Cad_prediction_Accurary()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        #self.writer.add_scalar('val/Clus_Acc_explict', Clus_Acc_explict, epoch)
        self.writer.add_scalar('val/Cad_predict_Acc', Cad_predict_Acc, epoch)
        self.writer.add_scalar('val/pixel_acc', pixel_acc, epoch)

        #Acc_value = self.evaluator_class.Accuracy()
        #Class_Acc_value = self.evaluator_class.Accuracy_Class()
        #classification_report_data = self.evaluator_class.sklearn_classification_report()
        #self.writer.add_scalar('val/Acc', Acc_value, epoch)
        #self.writer.add_scalar('val/Acc_Class', Class_Acc_value, epoch)
        #self.writer.add_text('Classicification_Report', classification_report_data, epoch)

        val_results_txt = 'Validation:\n' + \
            '[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0]) + \
            'Cad_Predict_Acc:{}, mIoU:{}, pixel_accuracy:{}\n'.format(Cad_predict_Acc, mIoU,pixel_acc) + \
            'Loss: %.3f' % test_loss

        print(val_results_txt)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        #print("Clus_Acc_explict:{}, Clus_Acc_implict:{}, mIoU:{}".format(Clus_Acc_explict, Clus_Acc_implict, mIoU))
        print("Cad_Predict_Acc:{}, mIoU:{}, pixel_accuracy:{}".format(Cad_predict_Acc, mIoU,pixel_acc))
        #print("Classification- Acc_value:{}, Class_Acc_value:{}".format(Acc_value, Class_Acc_value))
        print('Loss: %.3f' % test_loss)
        

        #self.writer.add_text('Validation', val_results_txt, epoch)

        #print(classification_report_data)

        new_pred = mIoU
        is_best = False
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            
        self.saver.save_checkpoint({
                'epoch': epoch + 1,
                #'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def test_batch(self, epoch):
        print('\n===================={}th Test =========================='.format(epoch))
        start_time = time.time()
        
        if self.args.dataset.startswith('samsung_SEM_BE'):
            self.evaluator = Evaluator_for_BE(2)
        else:
            self.evaluator = Evaluator(self.nclass, type='test')

        #self.model.eval()
        self.model.train()
        self.evaluator.reset()
        #tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tqdm(self.val_loader)):
            image, target = sample['image'], sample['label']
       
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if self.args.dataset.startswith('samsung_SEM_BE'):
                    output = torch.sigmoid(output)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            
            end_time = time.time()
            global_step = i
            self.summary.visualize_image_for_test(self.writer, self.args.dataset, image, target, output, global_step)
            
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            
            if self.args.dataset.startswith('samsung_SEM_BE'):
                pass
            else:
                pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            self.evaluator.add_batch_Clusters_Accurary(target, pred)

            # Fast test during the training
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            Acc = self.evaluator.Pixel_Accuracy()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            Clus_Acc_explict, Clus_Acc_implict = self.evaluator.Clusters_Accurary(target, pred)
            self.evaluator.show_confusion_matrix()
            #FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            self.writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('test/mIoU', mIoU, epoch)
            self.writer.add_scalar('test/Acc', Acc, epoch)
            self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
            #self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            
            print('NumImages: %5d' % (self.args.batch_size))
            print("Acc:{}, Acc_class:{}, mIoU:{}".format(Acc, Acc_class, mIoU))
            print("Clus_Acc_explict:{}, Clus_Acc_implict:{}".format(Clus_Acc_explict, Clus_Acc_implict))
            print('Loss: %.3f' % test_loss)
            print("Elaped Time: %s seconds" % ( end_time - start_time))
            break


 

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
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
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
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
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
