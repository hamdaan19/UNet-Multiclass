import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

MODEL_PATH = 'unet_cityscapes_CE.pth.tar'
LOAD_MODEL = False
ROOT_DIR = 'datasets/cityscapes'
IMG_HEIGHT = 110
IMG_WIDTH = 220
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 5

def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data): 
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
    
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        

def main():
    global epoch
    epoch = 0
    LOSS_VALS = []
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    ])

    train_set = get_cityscapes_data(
        split='train',
        mode='fine',
        relabelled=True,
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
    )

    print('Data Loaded Successfully!')

    unet = UNET(in_channels=3, classes=19).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)

    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    

    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')

        #taking the initial value of loss
        if e == 0:
            for batch in train_set:
                X, y = batch
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = unet(X)
                loss = loss_function(preds, y)
                LOSS_VALS.append(loss.item())
                print("Initial loss value recorded.")
                break

        loss_val = train_function(train_set, unet, optimizer, loss_function, DEVICE)
        LOSS_VALS.append(loss_val) 
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        print("Epoch completed and model successfully saved!")



if __name__ == '__main__':
    main()