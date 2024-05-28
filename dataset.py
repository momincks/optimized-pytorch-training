import os
import glob
import threading
from dataclasses import dataclass
from typing import Any
import PIL.Image
import numpy as np
import PIL
import torch


# subject to change
@dataclass
class SingleData:
    path:str=None
    image:np.ndarray=None
    label:Any=None

class Dataset(torch.utils.data.Dataset):

    '''
    in this example, we will be training image classification model.
    data structured as below:
    data_root:
        -> none:
            -> 1.jpg
            -> 2.jpg ...
        -> dog:
            -> 1.jpg
            -> 2.jpg ...
        -> cat:
            -> 1.jpg
            -> 2.jpg ...        
    '''

    def __init__(self, data_root, augment_data:bool=True):

        self.data_root = data_root
        self.augment_data = augment_data
        self.data_list = list()

        # subject to change
        image_paths = glob.glob(os.path.join(data_root, "*.jpg"))
        label_paths = image_paths.copy() # ensure sequence match with image_paths
        
        threading.Thread(target=self._preprocess, args=(image_paths, label_paths), daemon=True, name="dataset_cache_data").start()

    # subject to change
    @property
    def label_class():
        return {
            "none": 0,
            "dog": 1,
            "cat": 2
            }


    def _preprocess(self, image_paths:list=None, label_paths:list=None):
        # subject to change
        for image_path, label_path in zip(image_paths, label_paths):
            data = SingleData()
            data.image = PIL.Image.open(image_path).copy()
            data.label = self.label_class[os.path.split(label_path)[-1]] # get folder name and map into integer
            self.data_list.append(data)
    

    def __len__(self):
        return len(self.data_list)
    

    def _augment(self, data:SingleData):

        if not self.augment_data:
            return data
        
        image, label = data.image, data.label
        # subject to change
        # some augmentations
        data.image, data.label = image, label
        return data
    

    def __getitem__(self, index):
        return self._augment(self.data_list[index])