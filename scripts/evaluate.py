import torch
from model import UNET
from utils import *
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

ROOT_DIR_CITYSCAPES = 'datasets/cityscapes'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 600

MODEL_PATH = "YOUR-MODEL-PATH-WHICH-NEEDS-TO-BE-EVALUATED"

EVAL = False 
PLOT_LOSS = False

def save_predictions(data, model):    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):

            X, y, s = batch # here 's' is the name of the file stored in the root directory
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

            utils.save_as_images(pred_labels, location, name, multiclass=True)                

def evaluate(path):
    T = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.NEAREST)
    ])

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
    save_predictions(val_set, net)

def plot_losses(path):
    checkpoint = torch.load(path)
    losses = checkpoint['loss_values']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch))

    plt.plot(epoch_list, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epoch/s")
    plt.show()

if __name__ == '__main__':
    if EVAL:
        evaluate(MODEL_PATH)
    if PLOT_LOSS:
        plot_losses(MODEL_PATH)