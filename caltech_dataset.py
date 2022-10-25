from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import pandas as pd

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        path = 'Caltech101/' + split + '.txt';
        print("opening:", path)
        lines = []
        with open(path, 'r') as f:
            lines = f.readlines()
        
        
        label = -1
        str_label = '' 
        for line in lines:
            if line.startswith('BACKGROUND'):
                continue
            cur_label = line.split("/")[0]
            if cur_label != str_label:
                label += 1
                str_label = cur_label
            path = root +  '/' + line.strip()
            img = pil_loader(path)
            #print(f"appending: {cur_label}{label}")
            self.data.append((img, label))
        
        self.data = pd.DataFrame(self.data)
        self.data = self.data.sample(frac=1, replace=True)



    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data.iloc[index,0] , self.data.iloc[index,1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)
        return length
    
    def split(self, alpha):
      limit = alpha*len(self.data)
      return ([range(0, limit)], [range(limit, -1)])
      
     
