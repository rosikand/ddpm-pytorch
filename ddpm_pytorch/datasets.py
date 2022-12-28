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

    def __init__(self, data_set, normalize=True, resize=None, crop=None, neg_one_normalize=True, repeat_grayscale=False, dataset_multiplier=1):

        self.data_distribution = data_set
        self.normalize = normalize
        self.crop = crop
        self.neg_one_normalize = neg_one_normalize
        if self.neg_one_normalize and not self.normalize:
            raise ValueError("neg_one_normalize is set to True, but normalize is set to False. This is not allowed.")
        self.resize = resize 
        self.repeat_grayscale = repeat_grayscale

        if self.resize is not None and self.crop is not None:
            assert self.resize >= self.crop, "resize must be larger than crop"
        
        self.dataset_multiplier = dataset_multiplier


    def __getitem__(self, index):
        sample = self.data_distribution[index % len(self.data_distribution)]
        sample = torch.tensor(sample, dtype=torch.float)

        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0)
        
        assert len(sample.shape) == 3

        # ensure CHW format 
        if (sample.shape[0] != 3 and sample.shape[2] == 3) or (sample.shape[0] != 1 and sample.shape[2] == 1):
            sample = torch.movedim(sample, -1, 0)

        assert sample.shape[0] == 3 or sample.shape[0] == 1

        if self.repeat_grayscale:
            # makes grayscale images 3 channels to work with RGB models (e.g. u-net)
            if sample.shape[0] == 1:
                sample = torch.repeat_interleave(sample, 3, 0)
        

        if self.resize is not None:
            sample = torchvision.transforms.Resize(self.resize)(sample)

        if self.crop is not None:
            sample = torchvision.transforms.CenterCrop(self.crop)(sample)

        if self.normalize:
            sample = sample / 255.0
            
            if self.neg_one_normalize:
                sample = sample * 2 - 1
        
        return sample
        
    def __len__(self):
        return len(self.data_distribution) * self.dataset_multiplier



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
    def __init__(self, dirpath, resize=None, normalize=True, extension="jpg", crop=None, neg_one_normalize=True, repeat_graysclale=False, dataset_multiplier=1):
        self.dirpath = dirpath
        self.img_paths = glob(self.dirpath + "/*." + extension)
        self.resize = resize
        self.normalize = normalize
        self.crop = crop
        self.neg_one_normalize = neg_one_normalize
        if self.neg_one_normalize and not self.normalize:
            raise ValueError("neg_one_normalize is set to True, but normalize is set to False. This is not allowed.")
        self.repeat_grayscale = repeat_graysclale
        self.dataset_multiplier = dataset_multiplier
        
    def __getitem__(self, index):
        sample_file = self.img_paths[index % len(self.img_paths)]
        sample = ml.load_image(sample_file, resize=self.resize, normalize=self.normalize)

        sample = torch.tensor(sample, dtype=torch.float)

        if self.repeat_grayscale:
            # makes grayscale images 3 channels to work with RGB models (e.g. u-net)
            if sample.shape[0] == 1:
                sample = torch.repeat_interleave(sample, 3, 0)


        if self.neg_one_normalize and self.normalize:
            sample = sample * 2 - 1

        if self.crop is not None:
            sample = torchvision.transforms.CenterCrop(self.crop)(sample)

        return sample
        
    def __len__(self):
        return len(self.img_paths) * self.dataset_multiplier
        
