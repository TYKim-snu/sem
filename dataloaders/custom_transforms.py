import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from dataloaders.utils import encode_segmap
import cv2
from PIL import Image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img_CAD = sample['insert_CAD']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        img_CAD = np.array(img_CAD).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'insert_CAD': img_CAD,
                'label': mask}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ToTensor_for_SEM(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        
        mask = np.array(mask)
        mask = encode_segmap(mask)
        
        mask = np.array(mask).astype(np.float32)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

class ToTensor_for_BE(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img_CAD = sample['insert_CAD']
        mask = sample['label']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img_CAD = np.array(img_CAD).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        img_CAD = torch.from_numpy(img_CAD).float()
        mask = torch.from_numpy(mask).float()
        
        mask = mask[0,:,:].unsqueeze(0)
        mask = mask / 255.0
               
        return {'image': img,        
                'insert_CAD':img_CAD,
                'label': mask}

class ToTensor_for_BE_for_Test(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        
        return {'image': img,
                'label': None}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
 
        mask = np.array(mask).astype(np.float32)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #print('RandomHorizontalFlip',type(mask))
        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        random_rotate_idx = [0, 90, 180, 270]
        random_roate_seed = np.random.randint(0,len(random_rotate_idx))

        #rotate_degree = random.uniform(-1*self.degree, self.degree)
        #img = img.rotate(random_rotate_idx[random_roate_seed], Image.BILINEAR)
        #mask = mask.rotate(random_rotate_idx[random_roate_seed], Image.NEAREST)

        img = img.rotate(random_rotate_idx[random_roate_seed])
        mask = mask.rotate(random_rotate_idx[random_roate_seed])

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomColorJitter(transforms.ColorJitter):
    def __init__(self):
        super().__init__(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
    
    def __call__(self, sample):
        cad_img = sample['cad_image']
        sem_img = sample['sem_image']
        mask = sample['label']
        jittered_sem_image = super(RandomColorJitter,self).__call__(sem_img)
        #print('RandomColorJitter',type(mask))
        
        return {'cad_image': cad_img,
                'sem_image': jittered_sem_image,
                'label': mask}
      

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class CAD_SEM_merge_by_channel(object):
    def __call__(self, sample):
        cad_img = sample['cad_image']
        sem_img = sample['sem_image']
        mask = sample['label']
        
        cad_img = cad_img.convert('L')
        cad_img = np.array(cad_img)
        sem_img = sem_img.convert('L')
        sem_img = np.array(sem_img)

        cad_sem_img = np.dstack((sem_img, cad_img, sem_img))
        cad_sem_img = Image.fromarray(cad_sem_img)

        return {'image': cad_sem_img,
                'label': mask}

class CAD_SEM_merge(object):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, sample):
        cad_img = sample['cad_image']
        sem_img = sample['sem_image']
        mask = sample['label']

        cad_img = cad_img.convert('RGBA')
        cad_img = np.array(cad_img)
        
        # CAD에서 Green색만 남기기
        cad_img[:,:,0] = 0
        cad_img[:,:,2] = 0

        bg_color = np.array([0, 0, 0, 255]).astype('uint8')
        mask_for_merge = np.all(cad_img == bg_color, axis=2)
        cad_img[mask_for_merge] = [0, 0, 0, 0]
        
        sem_img = sem_img.convert('RGBA')
        sem_img = np.array(sem_img)
       
        output = cv2.addWeighted(cad_img, self.alpha, sem_img, 1-self.alpha, 0)
        
        cad_sem_img = cv2.cvtColor(output,cv2.COLOR_RGBA2RGB)
        cad_sem_img = Image.fromarray(cad_sem_img)
        #print('CAD_SEM_merge',type(mask))
        return {'image': cad_sem_img,
               'label': mask}

class RandomCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        #print('RandomCrop',type(mask))
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}