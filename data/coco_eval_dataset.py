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

FINE_TO_COARSE_PATH = 'fine_to_coarse_dict.pickle'

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class EvalCOCO(data.Dataset):
    def __init__(self, root, split, mode, res=128, transform_list=[], label=True, stuff=True, thing=False):
        self.root  = root 
        self.split = split
        self.mode  = mode
        self.res   = res 
        self.imdb  = self.load_imdb()
        self.stuff = stuff 
        self.thing = thing 
        self.label = label
        self.view  = -1

        self.fine_to_coarse = self._get_fine_to_coarse()

        # For test-time augmentation / robustness test. 
        self.transform_list = transform_list
        
    def load_imdb(self):
        # 1. Setup filelist
        # imdb = os.path.join(self.root, 'curated', '{}2017'.format(self.split), 'Coco164kFull_Stuff_Coarse_7.txt')
        imdb = os.listdir('/home/zhulifu/unsup_seg/STEGO-master/seg_dataset/coco_stuff/val2017')
        # imdb = tuple(open(imdb, "r"))
        imdb = sorted(list(set([id_[:-4] for id_ in imdb])))
        # imdb = [id_.rstrip() for id_ in imdb]
        # print(imdb)
        return imdb
    
    def __getitem__(self, index):
        image_id = self.imdb[index]
        img, lbl = self.load_data(image_id)

        return (index,) + self.transform_data(img, lbl, index)

    def load_data(self, image_id):
        """
        Labels are in unit8 format where class labels are in [0 - 181] and 255 is unlabeled.
        """
        N = len(self.imdb)
        image_path = os.path.join(self.root, '{}2017'.format(self.split), '{}.jpg'.format(image_id))
        label_path = os.path.join(self.root, '{}2017'.format(self.split), '{}.png'.format(image_id))

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        return image, label

    def transform_data(self, image, label, index, raw_image=False):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        label = TF.resize(label, self.res, Image.NEAREST)
        
        # 2. CenterCrop
        w, h = image.size
        left = int(round((w - self.res) / 2.))
        top  = int(round((h - self.res) / 2.))

        image = TF.crop(image, top, left, self.res, self.res)
        label = TF.crop(label, top, left, self.res, self.res)

        if raw_image:
            return image

        # 3. Transformation
        image = self._image_transform(image, self.mode)
        if not self.label:
            return (image, None)

        label = self._label_transform(label)

        return image, label


    def _get_fine_to_coarse(self):
        """
        Map fine label indexing to coarse label indexing. 
        """
        # with open(os.path.join(self.root, FINE_TO_COARSE_PATH), "rb") as dict_f:
        #     d = pickle.load(dict_f)
        # fine_to_coarse_dict      = d["fine_index_to_coarse_index"]
        # fine_to_coarse_dict[255] = -1
        # fine_to_coarse_map       = np.vectorize(lambda x: fine_to_coarse_dict[x]) # not in-place.

        # return fine_to_coarse_map
        return {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

    def _label_transform(self, label):
        """
        In COCO-Stuff, there are 91 Things and 91 Stuff. 
            91 Things (0-90)  => 12 superclasses (0-11)
            91 Stuff (91-181) => 15 superclasses (12-26)

        For [Stuff-15], which is the benchmark IIC uses, we only use 15 stuff superclasses.
        """
        label = np.array(label)
        # print(f'before {np.unique(label) = }')
        # label = self.fine_to_coarse(label)    # Map to superclass indexing.
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = np.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1
        # print(f'after {np.unique(label) = }')
        mask  = label >= 255 # Exclude unlabelled.
        label = coarse_label
        # Start from zero. 
        if self.stuff and not self.thing:
            label[mask] -= 12 # This makes all Things categories negative (ignored.)
        elif self.thing and not self.stuff:
            mask = label > 11 # This makes all Stuff categories negative (ignored.)
            label[mask] = -1
            
        # Tensor-fy
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
            trans_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))
        
        # Base transformation
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        return transforms.Compose(trans_list)
    
    def __len__(self):
        return len(self.imdb)
        

  
            
       
