import os
from PIL import Image 
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os

class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', data=None, mode='fine', relabelled=False, transform=None, eval=False):
        self.transform = transform
        self.relabelled = relabelled
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        if self.relabelled:
            # This block of code is used to prepare a list of all labelTrainIds rgb and 
            # ground truth images. Setting relabbelled=True is recommended. 

            self.label_path = os.path.join(os.getcwd(), root_dir+'/'+self.mode+'/'+self.split)
            self.rgb_path = os.path.join(os.getcwd(), root_dir+'/leftImg8bit/'+self.split)
            city_list = os.listdir(self.label_path)
            for city in city_list:
                temp = os.listdir(self.label_path+'/'+city)
                list_items = temp.copy()
        
                # 19-class label items being filtered
                for item in temp:
                    if not item.endswith('labelTrainIds.png', 0, len(item)):
                        list_items.remove(item)

                # defining paths
                list_items = ['/'+city+'/'+path for path in list_items]

                self.yLabel_list.extend(list_items)
                self.XImg_list.extend(
                    ['/'+city+'/'+path for path in os.listdir(self.rgb_path+'/'+city)]
                )
        else:
            self.data = datasets.Cityscapes(
             root_dir, split, mode=mode, target_type=target_type)
                
    def __len__(self):
        if self.relabelled: length = len(self.XImg_list)
        else: length = len(self.data)
        return length

    def __getitem__(self, index):
        if self.relabelled:
            image = Image.open(self.rgb_path+self.XImg_list[index])
            y = Image.open(self.label_path+self.yLabel_list[index])

            if self.transform is not None:
                image = self.transform(image)
                y = self.transform(y)

            image = transforms.ToTensor()(image)
            y = np.array(y)
            y = torch.from_numpy(y)
            
        else:
            batch = self.data[index]
            image, y = batch

            image = transforms.ToTensor()(image)
            y = transforms.ToTensor()(y)
            
            if self.transform is not None:
                image = self.transform(image)
                y = self.transform(y)


        y = y.type(torch.LongTensor)
        if self.eval:
            return image, y, self.XImg_list[index]
        else:
            return image, y
