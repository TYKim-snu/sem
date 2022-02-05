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

    def __init__(self, img_folder=None):
   
        super().__init__()
        
        print('CAD and genSEM and SEM Images from : {}'.format(img_folder))
        
        self._image_dir = img_folder
        self.images_list = self.sorted_aphanumeric(os.listdir(self._image_dir))
        #self.images_list = os.listdir(self._CAD_image_dir)
        print('Number of images : {:d}'.format(len(self.images_list)))

        self.im_ids = []
        self.images = []
        
        for ii, line in enumerate(self.images_list):
            _image = os.path.join(self._image_dir, line)
            assert os.path.isfile(_image)
            self.im_ids.append(line)
            self.images.append(_image)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _cad_img, _sem_img, _target_img, _img_name = self._make_img_gt_point_pair(index)
        sample = {'cad_image': _cad_img, 'sem_image': _sem_img, 'label': _target_img}
        result_dict = self.transform_val(sample)
        result_dict['image_name'] = _img_name
        return result_dict

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.images[index]).convert('RGB') #dummy
        _img_name = self.images_list[index]
        
        _cad_img = _img.crop((0, 0, 300, 300))
        _sem_img = _img.crop((300, 0, 900, 300))
        
        new_cad = Image.new('RGB', (600,300))
        new_cad.paste(_cad_img,(0,0))
        new_cad.paste(_cad_img,(300,0))

        return new_cad, _sem_img, _target, _img_name

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