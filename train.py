import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
#Hyperparameteres etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 256 
PIN_MEMORY = True 
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/all_4"#/vocalis_2"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/all_4"#/vocalis_2"

def train_fn(loader, model, optimizer, loss_fn, scaler, first_iterartion):
    loop = tqdm(loader)
    preds = torch.zeros(4,4,512,256)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        if batch_idx >= len(loader)-1:
            preds = torch.zeros(len(data),4,512,256)
        preds = preds.to(device = DEVICE)
        data = torch.cat((data,preds), 1)
        if(DEVICE == "cuda"):
            #forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets.squeeze().long())
            
            #backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds = predictions.detach_()
        


        #update tqdm loop
        loop.set_postfix(loss= loss.item())  

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Affine(translate_percent = 0.1,p = 0.25), #This leads to errors in combination with others
            A.OpticalDistortion(border_mode = cv2.BORDER_CONSTANT, shift_limit=0.7, distort_limit = 0.7, p = 0.5), #also play with shift_limit = 0.05 or distort_limit = 0.05
            A.Rotate(limit=60,border_mode = cv2.BORDER_CONSTANT, p=0.5), #Border Mode Constant for not duplicating the plica vocalis
            A.HorizontalFlip(p=0.5), #0.5
            A.VerticalFlip(p=0.25), #0.1
            A.RandomBrightnessContrast(contrast_limit = [-0.10, 0.6],p=0.5), #More in pos than neg, as brigther images are more possible to solve
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    """
    brightness_transform = A.Compose(
        [
            A.RandomBrightnessContrast(contrast_limit = [-6.0, 6.0],p=1.0), #[-0.10, 0.6],p=0.5), #More in pos than neg, as brigther images are more possible to solve
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    """


    model = UNET(in_channels=7, out_channels=4).to(DEVICE) #here out=x for more classes, was 1 to begin with
    loss_fn = nn.CrossEntropyLoss()     #-- at this point add nn.BCEWithLogitsLoss() for working single class
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
    )
    
    # Visualisation of the data augmentation:
    
    for i in range(0, 10):
        example = train_loader.dataset[i]
        image = example[0]
        mask = example[1]
        img_np = image.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
        cv2.imwrite("Example0.png", (img_np[0]*255).astype(np.uint8))
        cv2.imwrite("Example1.png", (img_np[1]*255).astype(np.uint8))
        cv2.imwrite("Example2.png", (img_np[2]*255).astype(np.uint8))
        cv2.imwrite("ExampleMask2.png", cv2.equalizeHist(mask_np.astype(np.uint8)))
    
    
    
    
    

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    if DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = "No Cuda = no GradScaler"
    


    #check_accuracy(val_loader, model, device= DEVICE, list=empty)
    for epoch in range(NUM_EPOCHS):
        if epoch == 0:
            first_iterartion = True
        else:
            first_iterartion = False
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler, first_iterartion)
    
        #save
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #check acc
        if epoch % 10 == 0 and epoch != 0:
            check_accuracy(val_loader, model, device= DEVICE)

        #print
        if epoch  == NUM_EPOCHS - 1:
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )  

if __name__ == "__main__":
    main()
