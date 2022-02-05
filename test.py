import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from modeling.deeplab import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary

from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders.datasets.samsung_CAD_BE_TEST import CADSegmentation
from dataloaders.custom_transforms import UnNormalize

    
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
from scipy.ndimage.morphology import binary_fill_holes

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from itertools import compress

class Tester(object):
    def __init__(self, args, CAD_image_dir=None, SEM_image_dir=None, list_txt=None, nclass=1):
        
        self.CAD_image_dir = CAD_image_dir
        self.SEM_image_dir = SEM_image_dir
        self.list_txt = list_txt
        #torch.cuda.set_device(args.gpu_ids)
        if args.useCPU is True:
            self.device = torch.device('cpu')
            print('Using device:',  self.device)
        else:
            self.device = torch.device('cuda:'+str(args.gpu_ids))
              # setting device on GPU if available, else CPU
            print('Using device:',  self.device)
            print('Available devices ', torch.cuda.device_count())
            print('Current cuda device ', args.gpu_ids)
      
            if self.device.type == 'cuda':
                print('GPU CARD {}'.format(str(args.gpu_ids)))
                print(torch.cuda.get_device_name(args.gpu_ids))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(args.gpu_ids)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(args.gpu_ids)/1024**3,1), 'GB')
         
        self.args = args
        self.nclass = nclass
        
        # Define Saver 
        self.saver = Saver(args, splt='test')
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        
        test_dataset = CADSegmentation(CAD_folder=CAD_image_dir, list_txt=self.list_txt)
        self.dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Define network
        model = DeepLab(args=args,
                        num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        freeze_bn=args.freeze_bn,
                        GroupNorm=args.GroupNorm)

        self.model = model
        if args.cuda:
            self.model = self.model.to(device=self.device)
        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location=self.device)

            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded trained Model '{}'"
                  .format(args.resume))
            args.start_epoch = 0

    def get_model(self):
        return self.model
    
    def test_batch(self, only_extract_result_img=False):
        self.model.eval()
        #self.model.freeze_dropout()
        
        tbar = tqdm(self.dataloader, desc='\r')
        num_img_tr = len(self.dataloader)
        for i, item in enumerate(tbar):
            image = item['image']
            image_name = item['image_name']
            if self.args.cuda:
                #image = image.cuda()
                image = image.to(device=self.device)
            with torch.no_grad():
                output = self.model(image)
                if self.args.dataset.startswith('samsung_CAD_BE'):
                    output = torch.sigmoid(output)
            
            global_step = i
           
            if 'BE' in self.args.dataset:
                self.postprocess_for_binary(image_name, image, output, global_step, 
                                            only_extract_result_img=only_extract_result_img)
            else:
                
                self.postprocess(image_name, image, output, global_step)
           
            pred = output.data.cpu().numpy() 
            if self.args.dataset.startswith('samsung_CAD_BE'):
                pass
            else:
                pred = np.argmax(pred, axis=1)

    
    def postprocess_for_binary(self, image_name, image, output, global_step,
                                    only_extract_result_img = False):
        unorm = UnNormalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        image_height= image.size()[2]
        image_width = image.size()[3]
        
        save_image_path = os.path.join(self.saver.directory,'results')
        if not(os.path.isdir(save_image_path)):
            os.makedirs(os.path.join(save_image_path))

        for i, (image_, image_name_, output_) in enumerate(zip(image, image_name, output)):    
            cad_sem_image = Image.open(os.path.join(self.SEM_image_dir, image_name_)).convert('RGB')
            cad_sem_image = np.array(cad_sem_image)

            sem_image = Image.open(os.path.join(self.SEM_image_dir, image_name_)).convert('RGB')
            sem_image = np.array(sem_image)

            #pre_image = torch.cat((output_.clone().cpu().data, output_.clone().cpu().data, output_.clone().cpu().data),dim=0)
            pre_image = torch.cat(( torch.zeros_like(output_.clone().cpu().data), 
                                    torch.zeros_like(output_.clone().cpu().data), 
                                    output_.clone().cpu().data ),dim=0)
                                    
            if only_extract_result_img is True:
                #pre_image_for_save = output_[0].detach().cpu().numpy() * 255
                pre_image_for_save = pre_image.numpy() > 0.3 * 1.0
                pre_image_for_save = pre_image_for_save * 255
                pre_image_for_save = pre_image_for_save.astype(np.uint8)
                pre_image_for_save = np.transpose(pre_image_for_save, axes=[1,2,0])
                result_file_name = os.path.join(save_image_path, image_name_.split('.')[0]+'.jpg')
                #result_img = Image.fromarray(pre_image_for_save).convert('RGB')
                #result_img.save(result_file_name)

                transparent_map = np.ones((image_height,image_width,1), dtype=np.uint8) * 255
                input_image_for_save = np.dstack((cad_sem_image, transparent_map))
                sem_image = np.dstack((sem_image, transparent_map))
                pre_image_for_save = np.dstack((pre_image_for_save, transparent_map))

                bg_color = np.array([0, 0, 0, 255]).astype('uint8')
                mask = np.all(pre_image_for_save == bg_color, axis=2)
                pre_image_for_save[mask] = [0, 0, 0, 0]
                
                alpha = 1.0
                pre_image_for_save = cv2.addWeighted(sem_image, 
                                1.0,
                                pre_image_for_save,
                                1.0,
                                0)
                
                result_img = Image.fromarray(pre_image_for_save).convert('RGB')
                result_img.save(result_file_name)

                #self.show_result(save_image_path, image_name_, input_image_for_save, pre_image, pre_image_for_save)
            else:
                pre_image_for_save = output_[0].detach().cpu().numpy()
                
                pre_image_for_save_red = pre_image_for_save >= 0.90
                pre_image_for_save_green = (pre_image_for_save >= 0.80) # & (pre_image_for_save < 0.90)
                pre_image_for_save_blue = (pre_image_for_save >= 0.60) # & (pre_image_for_save < 0.80)
                pre_image_for_save_yellow = (pre_image_for_save >= 0.50) # & (pre_image_for_save < 0.70)
                pre_image_with_boxes = self.draw_box_hotspot_nms((pre_image_for_save_red,
                                                                pre_image_for_save_green,
                                                                pre_image_for_save_blue,
                                                                pre_image_for_save_yellow))
                
                pre_image_for_save = pre_image_with_boxes.astype('uint8')
    
                transparent_map = np.ones((image_height,image_width,1), dtype=np.uint8) * 255
                input_image_for_save = np.dstack((cad_sem_image, transparent_map))
                sem_image = np.dstack((sem_image, transparent_map))
                pre_image_for_save = np.dstack((pre_image_for_save, transparent_map))

                bg_color = np.array([0, 0, 0, 255]).astype('uint8')
                mask = np.all(pre_image_for_save == bg_color, axis=2)
                pre_image_for_save[mask] = [0, 0, 0, 0]
                
                alpha = 1.0
                pre_image_for_save = cv2.addWeighted(sem_image, 
                                1.0,
                                pre_image_for_save,
                                1.0,
                                0)

                self.show_result(save_image_path, image_name_, input_image_for_save, pre_image, pre_image_for_save)

    def postprocess(self, image_name, image, output, global_step):
        unorm = UnNormalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        image_height= image.size()[2]
        image_width = image.size()[3]
        Image_folder_path = os.path.dirname(self.image_dir)
        Image_folder_path = os.path.join(Image_folder_path, 'Image')

        save_image_path = os.path.join(self.saver.directory,'results')
        if not(os.path.isdir(save_image_path)):
            os.makedirs(os.path.join(save_image_path))
        
        for i, (image_, image_name_, output_) in enumerate(zip(image, image_name, output)):    

            sem_image = Image.open(os.path.join(Image_folder_path, image_name_)).convert('RGB')
            sem_image = np.array(sem_image)

            output_ = torch.nn.functional.softmax(output_,dim=0)
            output_ = output_[1:]
            pre_image_for_save = torch.max(output_, 0)[0].detach().cpu().numpy()
            
            pre_image_for_save_red = pre_image_for_save >= 0.90
            pre_image_for_save_green = (pre_image_for_save >= 0.80) & (pre_image_for_save < 0.90)
            pre_image_for_save_blue = (pre_image_for_save >= 0.70) & (pre_image_for_save < 0.80)
            pre_image_for_save_yellow = (pre_image_for_save >= 0.50) & (pre_image_for_save < 0.70)
            pre_image_with_boxes = self.draw_box_hotspot_nms((pre_image_for_save_red,
                                                            pre_image_for_save_green,
                                                            pre_image_for_save_blue,
                                                            pre_image_for_save_yellow))

            pre_image_for_save = pre_image_with_boxes.astype('uint8')
 
            transparent_map = np.ones((image_height,image_width,1), dtype=np.uint8) * 255
            sem_image = np.dstack((sem_image, transparent_map))
            pre_image_for_save = np.dstack((pre_image_for_save, transparent_map))

            bg_color = np.array([0, 0, 0, 255]).astype('uint8')
            mask = np.all(pre_image_for_save == bg_color, axis=2)
            pre_image_for_save[mask] = [0, 0, 0, 0]
            
            alpha = 1.0
            pre_image_for_save = cv2.addWeighted(sem_image, 
                            1.0,
                            pre_image_for_save,
                            1.0,
                            0)

            self.show_result(save_image_path, image_name_, pre_image_for_save, pre_image_for_save)
            
    def show_result(self, save_image_path, image_name, input_image, predict_image, ouput_image ):        
        num_image_file = len(os.listdir(save_image_path))
        ii = num_image_file + (0)
        result_file_name = os.path.join(save_image_path, image_name.split('.')[0]+'.jpg')
        #result_img = Image.new("RGB",(1024*2 + 2, 1024))
        #result_img.paste(im=Image.fromarray(input_image), box=(0, 0))
        #result_img.paste(im=Image.fromarray(ouput_image), box=(1026, 0))
        result_img = Image.fromarray(ouput_image).convert('RGB')
        result_img.save(result_file_name)

        input_img_Tensor = torch.from_numpy(np.transpose(input_image, axes=[2,0,1]))
        #predict_img_Tensor = torch.from_numpy(np.transpose(predict_image, axes=[2,0,1]))
        output_img_Tensor = torch.from_numpy(np.transpose(ouput_image, axes=[2,0,1]))
        self.writer.add_image('Input_Image', input_img_Tensor, ii)
        self.writer.add_image('Output_Image', predict_image, ii)
        self.writer.add_image('Output_Image_with_Box', output_img_Tensor, ii)

    def draw_box_hotspot(self, input_images):
        with_boxes = np.zeros((input_images[0].shape[0],input_images[0].shape[1],3))
        img_width = input_images[0].shape[0]
        img_height = input_images[0].shape[1]

        colors = ( (255,0,0), (0,255,0), (0,255,255), (250,218,94))
        past_coordi =[]
        for ii, (input_image, color) in enumerate(zip(input_images, colors)):
            input_image = binary_fill_holes(input_image.copy())
            contours_data = find_contours(input_image, 0.5)
            bounding_boxes = []
            for contour in contours_data:

                Xmean = np.mean(contour[:,0])
                Ymean = np.mean(contour[:,1])
                
                Xmin = np.max([Xmean - 20, 1])
                Xmax = np.min([Xmean + 20, img_width-1])
                Ymin = np.max([Ymean - 20, 1])
                Ymax = np.min([Ymean + 20, img_height-1])
                
                box = [Xmin, Xmax, Ymin, Ymax]
                aug_box = [Xmin-1, Xmax+1, Ymin-1, Ymax+1]
                
                if len(past_coordi) == 0:
                    past_coordi.append([Xmean, Ymean])
                    r = [box[0],box[1],box[1],box[0], box[0]]
                    c = [box[3],box[3],box[2],box[2], box[3]]
                    rr, cc = polygon_perimeter(r, c, with_boxes.shape)
                    with_boxes[rr, cc] = color
                    r = [aug_box[0],aug_box[1],aug_box[1],aug_box[0], aug_box[0]]
                    c = [aug_box[3],aug_box[3],aug_box[2],aug_box[2], aug_box[3]]
                    rr, cc = polygon_perimeter(r, c, with_boxes.shape)
                    with_boxes[rr, cc] = color

                    if ii == 0:
                        with_boxes = self.text_phantom(with_boxes,'>0.9', coordi=(Ymin, Xmin), color=color)
                    elif ii == 1:
                        with_boxes = self.text_phantom(with_boxes,'>0.8', coordi=(Ymin, Xmin), color=color)
                    elif ii == 2:
                        with_boxes = self.text_phantom(with_boxes,'>0.6', coordi=(Ymin, Xmin), color=color)
                    elif ii == 3:
                        with_boxes = self.text_phantom(with_boxes,'>0.4', coordi=(Ymin, Xmin), color=color)
                else:    
                    dist_count = 0
                    for p_coor in past_coordi:
                        dist = np.linalg.norm(np.array([Xmean, Ymean])-np.array(p_coor))
                        if dist <= 30:
                            dist_count += 1
                    if dist_count == 0:
                        r = [box[0],box[1],box[1],box[0], box[0]]
                        c = [box[3],box[3],box[2],box[2], box[3]]
                        rr, cc = polygon_perimeter(r, c, with_boxes.shape)
                        with_boxes[rr, cc] = color

                        r = [aug_box[0],aug_box[1],aug_box[1],aug_box[0], aug_box[0]]
                        c = [aug_box[3],aug_box[3],aug_box[2],aug_box[2], aug_box[3]]
                        rr, cc = polygon_perimeter(r, c, with_boxes.shape)
                        with_boxes[rr, cc] = color
                        
                        past_coordi.append([Xmean, Ymean])
                        
                        if ii == 0:
                            with_boxes = self.text_phantom(with_boxes,'>0.9', coordi=(Ymin, Xmin), color=color)
                        elif ii == 1:
                            with_boxes = self.text_phantom(with_boxes,'>0.8', coordi=(Ymin, Xmin), color=color)
                        elif ii == 2:
                            with_boxes = self.text_phantom(with_boxes,'>0.6', coordi=(Ymin, Xmin), color=color)
                        elif ii == 3:
                            with_boxes = self.text_phantom(with_boxes,'>0.4', coordi=(Ymin, Xmin), color=color)
        return with_boxes

    def text_phantom(self, img, text, size=10, coordi=(0,0), color=None):
        img = img.astype('uint8')
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", size)
        draw.text(coordi, text, color, font=font)
        return np.array(img)

    def draw_box_hotspot_nms(self, input_images):
        with_boxes = np.zeros((input_images[0].shape[0],input_images[0].shape[1],3))
        img_width = input_images[0].shape[0]
        img_height = input_images[0].shape[1]
        probs = [0.9, 0.8, 0.65, 0.5]
        colors = ( (255,0,0), (0,255,0), (0,255,255), (250,218,94))
        box_colors = []
        bounding_boxes = []
        probs_boxes = []
        for ii, (input_image, color) in enumerate(zip(input_images, colors)):
            input_image = binary_fill_holes(input_image.copy())
            contours_data = find_contours(input_image, 0.5)
            
            for contour in contours_data:
                Xmean = np.mean(contour[:,0])
                Ymean = np.mean(contour[:,1])
                
                Xmin = np.max([Xmean - 20, 1])
                Xmax = np.min([Xmean + 20, img_width-1])
                Ymin = np.max([Ymean - 20, 1])
                Ymax = np.min([Ymean + 20, img_height-1])
                
                box = [Xmin, Ymin, Xmax-Xmin, Ymax-Ymin]
                bounding_boxes.append(box)
                box_colors.append(color)
                probs_boxes.append(probs[ii])
                #aug_box = [Xmin-1, Xmax+1, Ymin-1, Ymax+1]

        nms_result = self.nms(np.array(bounding_boxes), np.array(probs_boxes), 0.3)
        draw_boxes = list(compress(bounding_boxes, nms_result))
        draw_boxes_color = list(compress(box_colors, nms_result))

        for d_box, b_color in zip(draw_boxes, draw_boxes_color):
            Xmin = d_box[0]
            Xmax = d_box[2] + Xmin
            Ymin = d_box[1]
            Ymax = d_box[3] + Ymin
            
            box = [Xmin, Xmax, Ymin, Ymax]
            aug_box = [Xmin-1, Xmax+1, Ymin-1, Ymax+1]
            
            r = [box[0],box[1],box[1],box[0], box[0]]
            c = [box[3],box[3],box[2],box[2], box[3]]
            rr, cc = polygon_perimeter(r, c, with_boxes.shape)
            with_boxes[rr, cc] = b_color

            r = [aug_box[0],aug_box[1],aug_box[1],aug_box[0], aug_box[0]]
            c = [aug_box[3],aug_box[3],aug_box[2],aug_box[2], aug_box[3]]
            rr, cc = polygon_perimeter(r, c, with_boxes.shape)
            with_boxes[rr, cc] = b_color
            
            if b_color is colors[0]:
                with_boxes = self.text_phantom(with_boxes,'>0.9', coordi=(Ymin, Xmin), color=b_color)
            elif b_color is colors[1]:
                with_boxes = self.text_phantom(with_boxes,'>0.8', coordi=(Ymin, Xmin), color=b_color)
            elif b_color is colors[2]:
                with_boxes = self.text_phantom(with_boxes,'>0.6', coordi=(Ymin, Xmin), color=b_color)
            elif b_color is colors[3]:
                with_boxes = self.text_phantom(with_boxes,'>0.5', coordi=(Ymin, Xmin), color=b_color)

        return with_boxes

    def nms(self, boxes, probs, threshold):
        """Non-Maximum supression.
        Args:
            boxes: array of [cx, cy, w, h] (center format)
            probs: array of probabilities
            threshold: two boxes are considered overlapping if their IOU is largher than
                this threshold
            form: 'center' or 'diagonal'
        Returns:
            keep: array of True or False.
        """

        order = probs.argsort()[::-1]
        keep = [True]*len(order)
        for i in range(len(order)-1):
            ovps = self.batch_iou(boxes[order[i+1:]], boxes[order[i]])
            for j, ov in enumerate(ovps):
                if ov > threshold:
                    keep[order[j+i+1]] = False
        return keep
    
    def iou(self, box1, box2):
        """Compute the Intersection-Over-Union of two given boxes.
        Args:
            box1: array of 4 elements [cx, cy, width, height].
            box2: same as above
        Returns:
            iou: a float number in range [0, 1]. iou of the two boxes.
        """

        lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
            max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
        if lr > 0:
            tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
                max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
            if tb > 0:
                intersection = tb*lr
                union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

            return intersection/union

        return 0
    
    def batch_iou(self, boxes, box):
        """Compute the Intersection-Over-Union of a batch of boxes with another
        box.
        Args:
            box1: 2D array of [cx, cy, width, height].
            box2: a single array of [cx, cy, width, height]
        Returns:
            ious: array of a float number in range [0, 1].
        """
        lr = np.maximum(
            np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
            np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
            0
        )
        tb = np.maximum(
            np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
            np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
            0
        )
        inter = lr*tb
        union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
        return inter/union