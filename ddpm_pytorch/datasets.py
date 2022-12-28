"""
File: datasets.py
------------------
Implements common dataset classes for training diffusion models. 
"""


import torch
from torch.utils.data import Dataset
import rsbox 
from rsbox import ml, misc
import torchvision 
from glob import glob



class DistDataset(Dataset):

    def __init__(self, data_set, normalize=True, crop=None, neg_one_normalize=True):

        self.data_distribution = data_set
        self.normalize = normalize
        self.crop = crop
        self.neg_one_normalize = neg_one_normalize
        if self.neg_one_normalize and not self.normalize:
            raise ValueError("neg_one_normalize is set to True, but normalize is set to False. This is not allowed.")
        
    def __getitem__(self, index):
        sample = self.data_distribution[index % len(self.data_distribution)]
        sample = torch.tensor(sample, dtype=torch.float)

        if self.normalize:
            sample = sample / 255.0
            
            if self.neg_one_normalize:
                sample = sample * 2 - 1
                
        if self.crop is not None:
            sample = torchvision.transforms.CenterCrop(self.crop)(sample)

        return sample
        
    def __len__(self):
        return len(self.data_distribution)



class StreamingDataset(Dataset):
    """
    Dataset class for streaming data from a directory. 
    Specify path to directory containing images. 
    Arguments:
        dirpath: path to directory containing images
        resize: tuple of (height, width) to resize images to. None means no resizing.
        normalize: whether to normalize images to [0, 1]
        extension: extension of images in directory
        crop: (int) size to center crop images to. None means no cropping.
    """
    def __init__(self, dirpath, resize=None, normalize=True, extension="jpg", crop=None, neg_one_normalize=True):
        self.dirpath = dirpath
        self.img_paths = glob(self.dirpath + "/*." + extension)
        self.resize = resize
        self.normalize = normalize
        self.crop = crop
        self.neg_one_normalize = neg_one_normalize
        if self.neg_one_normalize and not self.normalize:
            raise ValueError("neg_one_normalize is set to True, but normalize is set to False. This is not allowed.")

        
    def __getitem__(self, index):
        sample_file = self.img_paths[index % len(self.img_paths)]
        sample = ml.load_image(sample_file, resize=self.resize, normalize=self.normalize)

        sample = torch.tensor(sample, dtype=torch.float)


        if self.neg_one_normalize and self.normalize:
            sample = sample * 2 - 1

        if self.crop is not None:
            sample = torchvision.transforms.CenterCrop(self.crop)(sample)

        return sample
        
    def __len__(self):
        return len(self.img_paths)
        
