import os 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
from data.custom_transforms import *

# We remove ignore classes. 
FINE_DICT = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:2, 10:3, 11:4, 12:5, 13:6,
             14:7, 15:8, 16:9, 17:10, 18:11, 19:12, 20:13, 21:14, 22:15, 23:16, 24:17, 25:18, 26:19,
             27:20, 28:21, 29:22, 30:23, 31:24, 32:25, 33:26, -1:-1}

COARSE_DICT = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:0, 9:0, 10:0, 11:1, 12:1, 13:1,
               14:1, 15:1, 16:1, 17:2, 18:2, 19:2, 20:2, 21:3, 22:3, 23:4, 24:5, 25:5, 26:6,
               27:6, 28:6, 29:6, 30:6, 31:6, 32:6, 33:6, -1:-1}
class TrainCityscapes(data.Dataset):
    def __init__(self, root, labeldir, mode, split='train', res1=320, res2=640, inv_list=[], eqv_list=[], scale=(0.5, 1), tar_res=40):
        self.root  = root 
        self.split = split
        self.res1  = res1
        self.res2  = res2  
        self.tar_res = tar_res
        self.mode  = mode
        self.scale = scale 
        self.view  = -1

        assert split == 'train', 'split should be [train].'
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir

        self.imdb = self.load_imdb()
        self.lbdb = self.load_lbdb()
        self.reshuffle() 
        LABEL_DICT = FINE_DICT
        self.cityscape_labelmap = np.vectorize(lambda x: LABEL_DICT[x])

    def load_imdb(self):
        imdb = []
        for folder in ['test', 'train', 'val']:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    image_path = os.path.join(self.root, 'leftImg8bit', folder, city, fname)
                    imdb.append(image_path)

        return imdb
    
    def load_lbdb(self):
        lbdb = []
        for folder in ['test', 'train', 'val']:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    label_name = fname.split('leftImg8bit.png')[0] + 'gtFine_labelIds.png'
                    label_path = os.path.join(self.root, 'gtFine', folder, city, label_name)
                    lbdb.append(label_path)

        return lbdb
    
    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        ipath = self.imdb[index]
        image = Image.open(ipath).convert('RGB')
        image = self.transform_image(index, image)
        label = self.transform_label(index)
        
        return (index, ) + image + label
    

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.imdb))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()


    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                # image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image)

            if self.mode == 'baseline_train':
                return (image1, )
            
            # image2 = TF.resize(image, self.res1, Image.BILINEAR)
            # image2 = self.transform_inv(index, image2, 1)
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image



    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image


    def init_transforms(self):
        N = len(self.imdb)
        
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, tar_res=self.tar_res, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1).unsqueeze_(0).unsqueeze_(0)
            label2 = label2.view(X2, X2).unsqueeze_(0).unsqueeze_(0)

            # size = self.tar_res
            # label1 = F.interpolate(label1.float(), (size, size), mode='nearest').long()[0,0]
            # label2 = F.interpolate(label2.float(), (size, size), mode='nearest').long()[0,0]

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X2, X2).unsqueeze_(0).unsqueeze_(0)
            size = self.tar_res
            label1 = F.interpolate(label1.float(), (size, size), mode='nearest').long()[0,0]

            return (label1, )
        
        elif self.mode == 'linear_train':
            path =  self.lbdb[index]
            label = Image.open(path).convert('L')
            size = self.tar_res
            label = label.resize((size, size), resample=Image.Resampling.NEAREST)
            # transforms.ToTensor()
            label = np.asarray(label)
            label = self.cityscape_labelmap(label)  
            label = torch.LongTensor(label)
            label[label < 0] = 27
            return (label, label)
        
        return (None, )


    def __len__(self):
        return len(self.imdb)
        

  
            
       

class TrainCityscapesRAW(data.Dataset):
    def __init__(self, root, labeldir, mode, split='train', res=320, res1=320, res2=640, inv_list=[], eqv_list=[], scale=(0.5, 1), tar_res=40):
        self.root  = root 
        self.split = split
        self.res1  = res1
        self.res2  = res2  
        self.tar_res = tar_res
        self.mode  = mode
        self.scale = scale 
        self.view  = -1

        assert split == 'train', 'split should be [train].'
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir

        self.imdb = self.load_imdb()
        self.lbdb = self.load_lbdb()
        self.reshuffle() 

    def load_imdb(self):
        imdb = []
        for folder in ['test', 'train', 'val']:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    image_path = os.path.join(self.root, 'leftImg8bit', folder, city, fname)
                    imdb.append(image_path)

        return imdb
    
    def load_lbdb(self):
        lbdb = []
        for folder in ['test', 'train', 'val']:
            for city in os.listdir(os.path.join(self.root, 'leftImg8bit', folder)):
                for fname in os.listdir(os.path.join(self.root, 'leftImg8bit', folder, city)):
                    label_name = fname.split('leftImg8bit.png')[0] + 'gtFine_color.png'
                    label_path = os.path.join(self.root, 'gtFine', folder, city, label_name)
                    lbdb.append(label_path)

        return lbdb

    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        ipath = self.imdb[index]
        image = Image.open(ipath).convert('RGB')
        image = self.transform_image(index, image)
        label = self.transform_label(index)
        
        return (index, ) + image + label
    

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.imdb))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()


    def transform_image(self, index, image):
        # Base transform
        # image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                # image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_inv(index, image, 1)
                # image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image)

            if self.mode == 'baseline_train':
                return (image1, )
            
            # image2 = TF.resize(image, self.res1, Image.BILINEAR)
            # image2 = self.transform_inv(index, image2, 1)
            image2 = self.transform_inv(index, image, 1)
            # image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image



    def transform_eqv(self, indice, image):
        # if 'random_crop' in self.eqv_list:
        #     image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image


    def init_transforms(self):
        N = len(self.imdb)
        
        # Base transform.
        # self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        # self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, tar_res=self.tar_res, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1).unsqueeze_(0).unsqueeze_(0)
            label2 = label2.view(X2, X2).unsqueeze_(0).unsqueeze_(0)

            size = self.tar_res
            label1 = F.interpolate(label1.float(), (size, size), mode='nearest').long()[0,0]
            label2 = F.interpolate(label2.float(), (size, size), mode='nearest').long()[0,0]

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X2, X2).unsqueeze_(0).unsqueeze_(0)
            size = self.tar_res
            label1 = F.interpolate(label1.float(), (size, size), mode='nearest').long()[0,0]

            return (label1, )

        elif 'linear_train' in self.mode:
            path =  self.lbdb[index]
            label = Image.open(path).convert('L')
            size = self.tar_res
            label = label.resize((size, size))
            # transforms.ToTensor()
            label = torch.LongTensor(np.asarray(label))
            return (label, label)

        return (None, )


    def __len__(self):
        return len(self.imdb)
        

  
            
       
