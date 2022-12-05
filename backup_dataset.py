import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

class VocDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_images = 3,transform=None):#transform_brightness=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #self.transform_brightness = transform_brightness
        self.images = sorted(os.listdir(image_dir))
        self.num_images = num_images
    
    def __len__(self):
        return len(self.images) - (self.num_images-1)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) 

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            #augmentations = self.transform_brightness(image=image)
            #imgFinal = augmentations["image"]
        
        """
        img_np = image.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()
        cv2.imwrite("Example0.png", (img_np[0]*255).astype(np.uint8))
        cv2.imwrite("Example1.png", (img_np[1]*255).astype(np.uint8))
        cv2.imwrite("Example2.png", (img_np[2]*255).astype(np.uint8))
        cv2.imwrite("ExampleMask2.png", cv2.equalizeHist(mask_np.astype(np.uint8)))
        """
        return image, mask