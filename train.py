import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
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
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 512 #256
IMAGE_WIDTH = 256 #128
PIN_MEMORY = True #makes transfer to GPU faster, so unnecessary right now
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/all_4"#/vocalis_2"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/all_4"#/vocalis_2"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
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

        #update tqdm loop
        loop.set_postfix(loss= loss.item())  

def main():
    train_transform = A.Compose(
        [
            # TODO:
            # helligkeit und affine translation und optical distortion
            # val set flippen, skript
            # -> analyse spekulare reflektionen, wie lange (Frames), Eingangsdaten vergrößern-> "Mehrere auf einmal", laserpunkte aus m mittel || optical flow || temporales backfeeding
            # präsi
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.65), #1.0
            A.HorizontalFlip(p=0.35), #0.5
            A.VerticalFlip(p=0.1), #0.1
            A.RandomBrightnessContrast(p=0.2),
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
    #train_transform = None
    #val_transform = None

    model = UNET(in_channels=3, out_channels=4).to(DEVICE) #here out=x for more classes, was 1 to begin with
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
    """
    for i in range(1, 10):
        example = train_loader.dataset[i]
        img = example[0].cpu().detach().numpy()
        mask = example[1].cpu().detach().numpy()
        cv2.imshow("Example", img[0])
        cv2.imshow("ExampleMask", mask)
        cv2.waitKey(0)
    """
    

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    if DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = "No Cuda = no GradScaler"

    for epoch in range(NUM_EPOCHS):
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
    
        #save
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

            #check acc
        check_accuracy(val_loader, model, device= DEVICE)

            #print
        if epoch % 5 == 0:
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )  

if __name__ == "__main__":
    main()
