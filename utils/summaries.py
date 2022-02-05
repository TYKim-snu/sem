import os
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence, decode_segmap
import matplotlib.pyplot as plt
from dataloaders.custom_transforms import UnNormalize
import cv2

from PIL import Image

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, 
                            image, target, output, image_name,
                            global_step, 
                            task='train', 
                            show_input_image=True,
                            validation_save=False,
                            val_ref_image_path=None):
        
        if task is 'validation':
            result_dir = os.path.join(self.directory, 'Validation_result_fig')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        
        unnorm = UnNormalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        
        if task is 'train':
            label = 'Training'
        elif task is 'validation':
            label = 'Validation'
        
        image = image[:1]
        target = target[:1]
        output = output[:1]
        image_name = image_name[:1]

        num_images = image.size()[0]
        image_width = image.size()[1]
        image_height = image.size()[2]
        
        show_images_num = 1

        pepe = Image.open('/data/1_data/0_Sample_Image/pepe.jpg')
        #pepe = numpy.asarray(pepe.resize((image_width, image_height)))
        
        if num_images is show_images_num:
            
            if task is 'train':
                figure1 = plt.figure(figsize=(4*3, 4*show_images_num))
                for fig in range(show_images_num):
                    figure1.add_subplot(show_images_num, 3, 3 * fig + 1)
                    if show_input_image is True:
                        npimg = (unnorm(image[fig]).numpy() * 255).astype('int')
                        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                    else:
                        plt.imshow(pepe, interpolation='nearest')
                    #plt.axis('off')
                    
                    figure1.add_subplot(show_images_num, 3, 3 * fig + 2)
                    np_target = np.transpose((target[fig].numpy()), (1,2,0))
                    #np_target = (np_target > 0.3) * 1
                    np_target = np.dstack((np_target, np_target, np_target)) * 255
                    plt.imshow(np_target.astype(np.uint8), interpolation='nearest')
                    #plt.axis('off')
                    
                    figure1.add_subplot(show_images_num, 3, 3 * fig + 3)
                    np_output= np.transpose((output[fig].numpy()), (1,2,0))
                    #np_output = (np_output > 0.3) * 1
                    np_output = np.dstack((np_output, np_output, np_output)) * 255
                    plt.imshow(np_output.astype(np.uint8), interpolation='nearest')
                    #plt.axis('off')
                
                writer.add_figure(label, figure1, global_step)
            
            if task is 'validation':
                figure2 = plt.figure(figsize=(4*4, 4*show_images_num))
                for fig in range(show_images_num):
                    figure2.add_subplot(show_images_num, 4, 4 * fig + 1)
                    if show_input_image is True:
                        npimg = (unnorm(image[fig].cpu()).detach().numpy() * 255).astype('int')
                        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                    else:
                        plt.imshow(pepe, interpolation='nearest')
                    
                    figure2.add_subplot(show_images_num, 4, 4 * fig + 2)
                    if val_ref_image_path is not None:
                        cadsemimg = Image.open(os.path.join(val_ref_image_path, image_name[fig]))
                        cadsemimg = np.asarray(cadsemimg)
                        if show_input_image is True:
                            plt.imshow(cadsemimg, interpolation='nearest')
                        else:
                            plt.imshow(pepe, interpolation='nearest')
                    else:
                        plt.imshow(pepe, interpolation='nearest')

                    figure2.add_subplot(show_images_num, 4, 4 * fig + 3)
                    np_target = np.transpose((target[fig].cpu().detach().numpy()), (1,2,0))
                    #np_target = (np_target > 0.3) * 1
                    np_target = np.dstack((np_target, np_target, np_target)) * 255
                    plt.imshow(np_target.astype(np.uint8), interpolation='nearest')
                    #plt.axis('off')
                    
                    figure2.add_subplot(show_images_num, 4, 4 * fig + 4)
                    np_output = np.transpose((output[fig].cpu().detach().numpy()), (1,2,0))
                    np_output = (np_output > 0.3) * 1
                    np_output = np.dstack((np.zeros_like(np_output), np.zeros_like(np_output), np_output)) * 255

                    transparent_map = np.ones((np_output.shape[0], np_output.shape[1], 1), dtype=np.uint8) * 255
                    #print(transparent_map.shape)
                    #input_image_for_save = np.dstack((cadsemimg, transparent_map))
                    sem_image = np.dstack((cadsemimg, transparent_map))
                    np_output = np.dstack((np_output, transparent_map)).astype('uint8')

                    bg_color = np.array([0, 0, 0, 255]).astype('uint8')
                    mask = np.all(np_output == bg_color, axis=2)
                    np_output[mask] = [0, 0, 0, 0]
                    
                    alpha = 1.0
                    np_output = cv2.addWeighted(sem_image, 
                                    1.0,
                                    np_output,
                                    1.0,
                                    0)
               
                    plt.imshow(np_output.astype(np.uint8), interpolation='nearest')
                
                result_file_num = len(os.listdir(result_dir))
                plt.savefig(os.path.join(result_dir, str(result_file_num) + '.png'))
                
                writer.add_figure(label, figure2, global_step)                    #plt.axis('off')
                
        
        plt.close('all')
        #if validation_save is True:
        #    self.save_model_result(image, output, target, image_name, 'Validation_result_full',
        #                    val_ref_image_path=val_ref_image_path)

    def visualize_image_backup(self, writer, dataset, 
                            image, target, output, image_name,
                            global_step, 
                            task='train', 
                            show_input_image=False,
                            validation_save=False,
                            val_ref_image_path=None):
        
        unorm = UnNormalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        
        if task is 'train':
            Predicted_label = 'Predicted label (Training)'
            Groundtruth_label = 'Groundtruth label (Training)'
        elif task is 'validation':
            Predicted_label = 'Predicted label (Validation)'
            Groundtruth_label = 'Groundtruth label (Validation)'
        
        if show_input_image is True:
            if image.size()[0] == 2:
                grid_image = torch.cat([
                                    unorm(image[0].clone().cpu().data), 
                                    unorm(image[1].clone().cpu().data) 
                                    ],dim=2)
            elif image.size()[0] >= 3:
                grid_image = torch.cat([
                                    unorm(image[0].clone().cpu().data), 
                                    unorm(image[1].clone().cpu().data), 
                                    unorm(image[2].clone().cpu().data)
                                    ],dim=2)
            else:
                grid_image = unorm(image[0].clone().cpu().data)
            
            writer.add_image('Image', grid_image, global_step)
            
        if 'BE' in dataset:
            expend_output1 = torch.cat((output[0].clone().cpu().data, output[0].clone().cpu().data,output[0].clone().cpu().data),dim=0)
            expend_output2 = torch.cat((output[1].clone().cpu().data, output[1].clone().cpu().data,output[1].clone().cpu().data),dim=0)
            expend_output3 = torch.cat((output[2].clone().cpu().data, output[2].clone().cpu().data,output[2].clone().cpu().data),dim=0)
            output_result = torch.cat((expend_output1, expend_output2, expend_output3), dim=2)
            writer.add_image(Predicted_label, output_result, global_step)
            
            expend_target1 = torch.cat((target[0].clone().cpu().data, target[0].clone().cpu().data, target[0].clone().cpu().data),dim=0)
            expend_target2 = torch.cat((target[1].clone().cpu().data, target[1].clone().cpu().data, target[1].clone().cpu().data),dim=0)
            expend_target3 = torch.cat((target[2].clone().cpu().data, target[2].clone().cpu().data, target[2].clone().cpu().data),dim=0)
            target_result = torch.cat((expend_target1, expend_target2, expend_target3), dim=2)
            writer.add_image(Groundtruth_label, target_result, global_step)
            
            if validation_save is True:
                self.save_model_result(image, output, target, image_name, 'Validation_result',
                                        val_ref_image_path=val_ref_image_path)

        else:
            grid_image2 = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image(Predicted_label, grid_image2, global_step)
            
            grid_image3 = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image(Groundtruth_label, grid_image3, global_step)

    
    def save_model_result(self, image, output, gt_image, image_name, save_folder, val_ref_image_path=None):
        unorm = UnNormalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))

        image_height= image.size()[2]
        image_width = image.size()[3]
            
        for i, (image_, image_name_, output_, gt_image_) in enumerate(zip(image, image_name, output, gt_image)):
            save_image_path = os.path.join(self.directory, save_folder)
            
            if not(os.path.isdir(save_image_path)):
                os.makedirs(os.path.join(save_image_path))
            
            if val_ref_image_path is None:
                image_for_save = unorm(image_.clone().cpu().data)
                image_for_save = np.transpose(image_for_save.numpy(), axes=[1,2,0])
            else:
                image_for_save = np.array(Image.open(os.path.join(val_ref_image_path,image_name_)))
            
            pre_image_for_save = torch.cat((output_.clone().cpu().data, output_.clone().cpu().data, output_.clone().cpu().data),dim=0)
            pre_image_for_save =np.transpose(pre_image_for_save.numpy(), axes=[1,2,0])
            gt_image_for_save = torch.cat((gt_image_.clone().cpu().data, gt_image_.clone().cpu().data, gt_image_.clone().cpu().data),dim=0)
            gt_image_for_save = np.transpose(gt_image_for_save.numpy(), axes=[1,2,0])
            
            image_for_save = image_for_save.astype('uint8')
            pre_image_for_save = (pre_image_for_save * 255).astype('uint8')
            gt_image_for_save = (gt_image_for_save * 255).astype('uint8')

            transparent_map = np.ones((image_height,image_width,1), dtype=np.uint8) * 255
            image_for_save = np.dstack((image_for_save,transparent_map))
            gt_image_for_save = np.dstack((gt_image_for_save,transparent_map))
            pre_image_for_save = np.dstack((pre_image_for_save,transparent_map))
            
            bg_color = np.array([0, 0, 0, 255]).astype('uint8')
            mask = np.all(pre_image_for_save == bg_color, axis=2)
            pre_image_for_save[mask] = [0, 0, 0, 0]
            mask = np.all(gt_image_for_save == bg_color, axis=2)
            gt_image_for_save[mask] = [0, 0, 0, 0]
            
            alpha = 1.0
            pre_image_for_save = cv2.addWeighted(pre_image_for_save, 
                            alpha,
                            image_for_save,
                            0.8,
                            0)
            gt_image_for_save = cv2.addWeighted(gt_image_for_save, 
                alpha,
                image_for_save,
                0.8,
                0)

            result_file_name = os.path.join(save_image_path, image_name_.split('.')[0]+'.jpg')
            result_img = Image.new("RGB",(image_width*3 + 4, image_height))
            result_img.paste(im=Image.fromarray(image_for_save), box=(0, 0))
            result_img.paste(im=Image.fromarray(pre_image_for_save), box=(image_width + 2, 0))
            result_img.paste(im=Image.fromarray(gt_image_for_save), box=(2 * image_width + 4, 0))
            result_img.save(result_file_name)

    