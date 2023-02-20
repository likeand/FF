import os 
import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import json
import random 
import cv2
import pickle


class TrainMedical(data.Dataset):
    def __init__(self, root, res=128, transform_list=[], label=True):
        self.root  = root 
        self.res   = res 
        self.label = label
        self.long_image = False
        # For test-time augmentation / robustness test. 
        self.transform_list = transform_list
        self.mode = 'test'
        self.imdb = self.load_imdb()
        
    def load_imdb(self):
        imdb = []

        for fname in os.listdir(os.path.join(self.root, 'imgs')):
            image_path = os.path.join(self.root, 'imgs', fname)
            lname = fname
            label_path = os.path.join(self.root, 'masks',  lname)
            imdb.append((image_path, label_path))

        return imdb
        
    def __getitem__(self, index):
        impath, gtpath = self.imdb[index]

        image = Image.open(impath).convert('RGB')
        label = Image.open(gtpath) if self.label else None 

        return (index,) + self.transform_data(image, label, index)


    def transform_data(self, image, label, index):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        
        # 2. CenterCrop
        if not self.long_image:
            w, h = image.size
            left = int(round((w - self.res) / 2.))
            top  = int(round((h - self.res) / 2.))

            image = TF.crop(image, top, left, self.res, self.res)
            
        # 3. Transformation
        image = self._image_transform(image, self.mode)
        if not self.label:
            return (image, None)

        label = TF.resize(label, self.res, Image.NEAREST) 
        label = TF.crop(label, top, left, self.res, self.res) if not self.long_image else label
        label = self._label_transform(label)

        return image, label


    def _label_transform(self, label):
        label = np.array(label)
        label = torch.LongTensor(label)                            

        return label


    def _image_transform(self, image, mode):
        if self.mode == 'test':
            transform = self._get_data_transformation()
            return transform(image)
        else:
            raise NotImplementedError()


    def _get_data_transformation(self):
        trans_list = []
        if 'jitter' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8))
        if 'grey' in self.transform_list:
            trans_list.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.GaussianBlur((5, 5), (0.1, 2))], p=0.5))
        
        # Base transformation
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        return transforms.Compose(trans_list)
    
    def __len__(self):
        return len(self.imdb)
        
def getStat():
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    # stat = ([-0.8600419, -0.5789229, -0.5359389], [0.7740311, 0.8117282, 0.792767])
    stat = ([0.28805077, 0.32632148, 0.2854132], [0.17725338, 0.18182696, 0.17837301])
    if stat is not None:
        return stat

  