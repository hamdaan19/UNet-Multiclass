import torch
import os
from model import UNET
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from cityscapesscripts.helpers.labels import trainId2label as t2l

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

TEST_IMAGES_DIR_CARVANA = os.path.join(os.getcwd(), "datasets/carvana_dataset/test_unet_images")
TEST_MASKS_DIR_CARVANA = os.path.join(os.getcwd(), "datasets/carvana_dataset/test_unet_masks")
ROOT_DIR_CITYSCAPES = 'datasets/cityscapes'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 600

MODEL_PATH = 1
DATASET = 1

EVAL = False
SAVE_IMG = False 
PLOT_LOSS = True

dataset = {0: 'carvana', 1: 'cityscapes'}
model = {0: 'unet_carvana.pth.tar', 1: 'unet_cityscapes_CE.pth.tar'}


def remap_label(x):
    return t2l[x].id 

def compute_accuracy(data, model):
    correct = 0
    total = 0
    dice_score = 0
    accuracy = 0
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):
            if dataset[DATASET] == 'carvana':
                X, y = batch
                X, y = X.to(device), y.unsqueeze(1).to(device)
                predictions = model(X)     
                correct += (predictions == y).sum()
                total += torch.numel(predictions)
                dice_score += (2*(predictions*y).sum()) / ((predictions+y).sum())
                dice_score = dice_score/len(data)
                accuracy = correct/total * 100

                return [accuracy, dice_score]

            elif dataset[DATASET] == 'cityscapes':
                X, y, s = batch
                X, y = X.to(device), y.to(device)
                predictions = model(X) 
                
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                pred_labels = torch.argmax(predictions, dim=1) 
                pred_labels = pred_labels.float()

                # Remapping the labels
                pred_labels = pred_labels.to('cpu')
                pred_labels.apply_(lambda x: t2l[x].id)
                pred_labels = pred_labels.to(device)   

                # Resizing predicted images too original size
                pred_labels = transforms.Resize((1024, 2048))(pred_labels)             

                # Configure filename & location to save predictions as images
                s = str(s)
                pos = s.rfind('/', 0, len(s))
                name = s[pos+1:-18]  
                global location
                location = 'saved_images\multiclass_1'

                if SAVE_IMG == True:
                    save_as_images(pred_labels, location, name, multiclass=True)                

    

def eval_singleclass(path):
    T = A.Compose([
        A.Resize(height=160, width=240),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    val_data = get_carvana_data(
        TEST_IMAGES_DIR_CARVANA,
        TEST_MASKS_DIR_CARVANA,
        transforms=T
    )

    net = UNET(in_channels=3, classes=1).to(device)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{path} has been loaded and initialized')

    acc = compute_accuracy(val_data, net, device)
    print(f"Accuracy: {acc[0]:.4f} %")
    print(f"Dice Score: {acc[1]}")

def eval_multiclass(path):
    T = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    ])

    print('Loading Test Data')

    val_set = get_cityscapes_data(
        root_dir=ROOT_DIR_CITYSCAPES,
        split='val',
        mode='fine',
        relabelled=True,
        transforms=T,
        shuffle=True,
        eval=True
    )
 
    print('Data has been loaded!')

    net = UNET(in_channels=3, classes=19).to(device)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{path} has been loaded and initialized')
    compute_accuracy(val_set, net)
    #print(f'All images has been saved in {os.path.join(os.getcwd(), location)}!')


def plot_losses(path):
    checkpoint = torch.load(path)
    losses = checkpoint['loss_values']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch+3))

    plt.plot(epoch_list, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epoch/s")
    plt.show()

if __name__ == '__main__':
    if EVAL:
        if dataset[DATASET] == 'carvana':
            eval_singleclass(model[MODEL_PATH])
        elif dataset[DATASET] == 'cityscapes':
            eval_multiclass(model[MODEL_PATH])
    if PLOT_LOSS:
        plot_losses(model[MODEL_PATH])