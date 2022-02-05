from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

from dataloaders.utils import encode_segmap

class CADSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 1

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('samsung_CAD_BE_crop_256_NormalRatio_1'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        print(self._base_dir)
        self._CAD_image_dir = os.path.join(self._base_dir, 'Image')
        # self._CAD_image_dir = os.path.join(self._base_dir, 'Image')
        self._CAD2_image_dir = os.path.join(self._base_dir, 'CAD')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        
        #self._cat_dir = os.path.join(self._base_dir, 'HeatmapObject')
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Main')

        self.im_ids = []
        self.CAD_images = []
        self.CAD2_images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                
                _cad_image = os.path.join(self._CAD_image_dir, line)
                _cad2_image = os.path.join(self._CAD2_image_dir, line)
                _cat = os.path.join(self._cat_dir, line)
                assert os.path.isfile(_cad_image)
                assert os.path.isfile(_cad2_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.CAD_images.append(_cad_image)
                self.CAD2_images.append(_cad2_image)
                self.categories.append(_cat)

        assert (len(self.CAD_images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.CAD_images)))

    def __len__(self):
        return len(self.CAD_images)


    def __getitem__(self, index):
        _cad_img, _cad2_img, _target, _img_name = self._make_img_gt_point_pair(index)
        #print(_cad2_img.size)
        sample = {'image': _cad_img, 'insert_CAD':_cad2_img,'label': _target}

        for split in self.split:
            if split == "train":
                result = self.transform_tr(sample)
                result['image_name'] = _img_name
                return result
            
            elif split == 'val':
                result = self.transform_val(sample)
                result['image_name'] = _img_name
                return result 
            
            elif split == 'test':
                result = self.transform_val(sample)
                result['image_name'] = _img_name
                return result 

    def _make_img_gt_point_pair(self, index):
        #_img = Image.open(self.images[index]).convert('RGB')
        _cad_img = Image.open(self.CAD_images[index]).convert('RGB')
        #print(_cad_img.size)
        _cad2_img = Image.open(self.CAD2_images[index]).convert('RGB')
        # print(_cad2_img.size)
        _target = Image.open(self.categories[index]).convert('RGB')
        _img_name = self.im_ids[index]
        #print(self.categories[index])
        return _cad_img, _cad2_img, _target, _img_name

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomColorJitter(),
            #tr.CAD_SEM_merge(alpha=0.08),
            #tr.CAD_SEM_merge_by_channel(),
            #tr.RandomHorizontalFlip(),
            #tr.RandomRotate(),
            #tr.RandomCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
            tr.ToTensor_for_BE()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.CAD_SEM_merge(alpha=0.08),
            #tr.CAD_SEM_merge_by_channel(),
            #tr.RandomCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
            tr.ToTensor_for_BE()])

        return composed_transforms(sample)

    def __str__(self):
        return 'CAD DATA(split=' + str(self.split) + ')'
