from __future__ import print_function, division
import os
import re
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

from dataloaders.utils import encode_segmap

class SEMSegmentation(Dataset):

    def __init__(self, CAD_folder=None, SEM_folder=None):
   
        super().__init__()
        
        print('CAD Images from : {}'.format(CAD_folder))
        print('SEM Images from : {}'.format(SEM_folder))
        
        self._CAD_image_dir = CAD_folder
        self._SEM_image_dir = SEM_folder
        self.images_list = self.sorted_aphanumeric(os.listdir(self._CAD_image_dir))
        #self.images_list = os.listdir(self._CAD_image_dir)
        print('Number of images : {:d}'.format(len(self.images_list)))

        self.im_ids = []
        self.CAD_images = []
        self.SEM_images = []
   
        for ii, line in enumerate(self.images_list):
            _cad_image = os.path.join(self._CAD_image_dir, line)
            _sem_image = os.path.join(self._SEM_image_dir, line)
            assert os.path.isfile(_cad_image)
            assert os.path.isfile(_sem_image)
            self.im_ids.append(line)
            self.CAD_images.append(_cad_image)
            self.SEM_images.append(_sem_image)
            
        assert (len(self.CAD_images) == len(self.SEM_images))

    def __len__(self):
        return len(self.CAD_images)

    def __getitem__(self, index):
        _cad_img, _sem_img, _target_img, _img_name = self._make_img_gt_point_pair(index)
        sample = {'cad_image': _cad_img, 'sem_image': _sem_img, 'label': _target_img}
        result_dict = self.transform_val(sample)
        result_dict['image_name'] = _img_name
        return result_dict

    def _make_img_gt_point_pair(self, index):
        _cad_img = Image.open(self.CAD_images[index]).convert('RGB')
        _sem_img = Image.open(self.SEM_images[index]).convert('RGB')
        _target = Image.open(self.CAD_images[index]).convert('RGB') #dummy
        _img_name = self.images_list[index]
        return _cad_img, _sem_img, _target, _img_name

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.CAD_SEM_merge_by_channel(),
            tr.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
            tr.ToTensor_for_BE()])
        return composed_transforms(sample)

    def sorted_aphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)