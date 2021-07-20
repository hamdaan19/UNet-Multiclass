import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from datasets import CarvanaDataset, CityscapesDataset
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

def get_carvana_data(
    image_dir,
    mask_dir,
    batch_size=1,
    transforms=None,
    pin_memory=True
):      
    data = CarvanaDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transforms
    )

    data_loaded = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory
    )

    return data_loaded

def get_cityscapes_data(
    mode,
    split,
    relabelled,
    root_dir='datasets/cityscapes',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type, relabelled=relabelled, transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return data_loaded

def save_as_images(tensor_pred, folder, image_name, tensor_y=None, multiclass=False):
    if multiclass == False:
        utils.save_image(tensor_pred, f"{folder}/{image_name}_pred.png")
        if tensor_y is not None:
            utils.save_image(tensor_y, f"{folder}/{image_name}_true.png")
    elif multiclass == True:
        tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
        filename = f"{folder}\{image_name}.png"
        tensor_pred.save(filename)
        


        

    