import torch
import torchvision
import cv2
import numpy as np
from dataset import VocDataset
from torch.utils.data import DataLoader

BINARY = False
MULTI_IMAGE_INPUT = False

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_ds = VocDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    val_ds = VocDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def afterColor(img):
    result = np.zeros([3,512,256])
    for x in range(0, 512):
        for y in range(0, 256):
            value = img[x][y] -1
            if(value >= 0):
                result[int(value)][x][y] = 1
    return result


def checkNeighbours(list, masks, preds):
    i = list[0]
    x = list[1]
    y = list[2]

    masks[i][x][y] = 2
    counter = 0
    if(preds[i][x+1][y] == 1):
        counter+=1
    elif(preds[i][x-1][y] == 1):
        counter+=1
    elif(preds[i][x][y+1] == 1):
        counter+=1
    elif(preds[i][x][y-1] == 1):
        counter+=1
    elif(preds[i][x][y] == 1):
        counter+=1
    if counter > 0:
        masks = setNeighbours([i,x,y], masks)
        return True
    else:
        if masks[i][x+1][y] == 1:
            c =  checkNeighbours([i,x+1,y], masks,preds)
            if c:
                return c
        if masks[i][x-1][y] == 1:
            c =  checkNeighbours([i,x-1,y], masks,preds)
            if c:
                return c
        if masks[i][x][y+1] == 1:
            c =  checkNeighbours([i,x,y+1], masks,preds)
            if c:
                return c
        if masks[i][x][y-1] == 1:
            c =  checkNeighbours([i,x,y-1], masks,preds)
            if c:
                return c
        return False


def setNeighbours(list, masks):
    i = list[0]
    x = list[1]
    y = list[2]

    masks[i][x][y] = 0
    if masks[i][x+1][y] == 1 or masks[i][x+1][y] == 2:
        masks = setNeighbours([i,x+1,y], masks)
    if masks[i][x-1][y] == 1 or masks[i][x-1][y] == 2:
        masks = setNeighbours([i,x-1,y], masks)
    if masks[i][x][y+1] == 1 or masks[i][x][y+1] == 2:
        masks = setNeighbours([i,x,y+1], masks)
    if masks[i][x][y-1] == 1 or masks[i][x][y-1] == 2:
        masks = setNeighbours([i,x,y-1], masks)
    return masks



def refiningSoftmax(preds, color=False):
    #if not MULTI_IMAGE_INPUT:
    preds = preds.cpu().numpy()
    new_preds = []
    p = 0
    for i in range (0, len(preds)):
        result = np.zeros([512,256])
        for x in range(0, 512):
            for y in range(0, 256):
                values = [preds[i][0][x][y],preds[i][1][x][y],preds[i][2][x][y],preds[i][3][x][y]]
                index_max = max(range(len(values)), key=values.__getitem__)
                if(not index_max == 0):
                    p += 1
                result[x][y] = index_max
        if color == True:
            real_result = afterColor(result)
            new_preds.append(real_result)
            continue
        new_preds.append(result)
    new_preds = np.array(new_preds)
    new_preds = torch.from_numpy(new_preds)
    return new_preds
    """
    else:
        preds = preds.cpu().numpy()
        new_preds = []
        for i in range(0, len(preds)):
            #i = len(preds)-1
            result = np.zeros([512,256])
            for x in range(0, 512):
                for y in range(0, 256):
                    values = [preds[i][0][x][y],preds[i][1][x][y],preds[i][2][x][y],preds[i][3][x][y]]
                    index_max = max(range(len(values)), key=values.__getitem__)
                    result[x][y] = index_max
            new_preds.append(result)
        return torch.from_numpy(np.array(new_preds))
    """

def refine(laserMask, laserPred):
    masks = laserMask.cpu().numpy()
    preds = laserPred.cpu().numpy()
    hitCounter = 0
    totalCounter = 0
    result = 1.0
    for i in range (0, len(preds)):
        for x in range(0, 512):
            for y in range(0, 256):
                if(masks[i][x][y] == 1):
                    totalCounter += 1
                    c = checkNeighbours([i,x,y], masks, preds)
                    if c:
                        hitCounter += 1
                        #masks = v
                    
                    y+=3
        """
        for x in range(0, 512):
            for y in range(0, 256):
                if(masks[i][x][y] == 1 or masks[i][x][y] == 2):
                    totalCounter += 1
                    masks = setNeighbours([i,x,y], laserMask)
        """
    result = hitCounter/totalCounter
    return result 

def check_accuracy(loader, model, device="cpu"):
    print("#######################")
    print("initiate accuarcy_check")
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    dice_score_glottis1 = 0
    dice_score_vocalis2 = 0
    dice_score_laserdots3 = 0
    hit_chance_laser = -1.0
    model.eval()

    if not MULTI_IMAGE_INPUT:
        if BINARY:
            with torch.no_grad():
                for img, mask in loader:
                        img = img.to(device)
                        mask = mask.to(device).unsqueeze(1)
                        preds = torch.sigmoid(model(img))
                        preds = (preds > 0.5).float()
                        num_correct += (preds == mask).sum()
                        num_pixels += torch.numel(preds)
                        dice_score += (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-8)

            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
            print(f"Dice score: {dice_score/len(loader)}")
        else:
            with torch.no_grad():
                number = -1
                final = len(loader)
                preds = torch.zeros(4,4,512,256)
                for img, mask in loader:
                    number +=1
                    if number%5 == 0:
                        print("Working on pic "+ str(number) +" / " +str(final))
                    # Getting the results

                    img = img.to(device)
                    mask = mask.to(device)
                    if number >= len(loader)-1:
                        preds = torch.zeros(len(img),4,512,256)
                    preds = preds.to(device)
                    img = torch.cat((img,preds), 1)
                    preds = torch.softmax(model(img), 1)
                    new_preds = refiningSoftmax(preds)

                    # Calculating the right pixels

                    mask = mask.cpu()
                    num_correct += (new_preds == mask).sum()
                    num_pixels += torch.numel(new_preds)

                    # Dice Scores

                    zero = torch.zeros_like(mask)
                    one = torch.ones_like(mask)

                    glottisMask = torch.where(mask != 1, zero, one)
                    vocalisMask = torch.where(mask != 2, zero, one)
                    laserMask = torch.where(mask != 3, zero, one)

                    glottisPred = torch.where(new_preds != 1, zero, one)
                    vocalisPred = torch.where(new_preds != 2, zero, one)
                    laserPred = torch.where(new_preds != 3, zero, one)

                    dice_score_glottis1 += (2 * (glottisPred * glottisMask).sum()) / ((glottisPred + glottisMask).sum() + 1e-8)
                    dice_score_vocalis2 += (2 * (vocalisPred * vocalisMask).sum()) / ((vocalisPred + vocalisMask).sum() + 1e-8)
                    dice_score_laserdots3 += (2 * (laserPred * laserMask).sum()) / ((laserPred + laserMask).sum() + 1e-8)

                    zwiSave = refine(laserMask, laserPred)
                    if hit_chance_laser == -1.0:
                        hit_chance_laser = zwiSave
                    else:
                        hit_chance_laser = (zwiSave + hit_chance_laser)/2

            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")

            print(f"Dice score for the glottis segmentation: {dice_score_glottis1/(len(loader))}")
            print(f"Dice score for the vocalis segmentation: {dice_score_vocalis2/(len(loader))}")
            print(f"Dice score for the laserdots segmentation: {dice_score_laserdots3/(len(loader))}")
            print(f"Laserdots Hit Chance: {hit_chance_laser}")
    else:
        with torch.no_grad():
            number = 0
            final = len(loader)
            for img, mask in loader:
                img_np = img.cpu().detach().numpy()
                """
                mask_np = mask.cpu().detach().numpy()
                cv2.imwrite("Example0.png", (img_np[0][0]*255).astype(np.uint8))
                cv2.imwrite("Example1.png", (img_np[1][0]*255).astype(np.uint8))
                cv2.imwrite("Example2.png", (img_np[2][0]*255).astype(np.uint8))
                cv2.imwrite("ExampleMask2.png", cv2.equalizeHist(mask_np.astype(np.uint8)))
                new_img_set = []
                new_img_set.append(img_np[0][2])
                new_img_set.append(img_np[1][2])
                new_img_set.append(img_np[2][2])
                new_img_set.append(img_np[3][2])
                new_img_set = np.array(new_img_set)
                new_img_set = torch.from_numpy(new_img_set)
                """
                number +=1
                #if(number > final/2):
                #    break
                if number%5 == 0:
                    print("Working on pic "+ str(number) +" / " +str(final))
                # Getting the results

                img = img.to(device)
                mask = mask.to(device)
                preds = torch.softmax(model(img), 1)
                new_preds = refiningSoftmax(preds)

                # Calculating the right pixels

                mask = mask.cpu()
                num_correct += (new_preds == mask).sum()
                num_pixels += torch.numel(new_preds)

                # Dice Scores

                zero = torch.zeros_like(mask)
                one = torch.ones_like(mask)

                glottisMask = torch.where(mask != 1, zero, one)
                vocalisMask = torch.where(mask != 2, zero, one)
                laserMask = torch.where(mask != 3, zero, one)

                glottisPred = torch.where(new_preds != 1, zero, one)
                vocalisPred = torch.where(new_preds != 2, zero, one)
                laserPred = torch.where(new_preds != 3, zero, one)

                dice_score_glottis1 += (2 * (glottisPred * glottisMask).sum()) / ((glottisPred + glottisMask).sum() + 1e-8)
                dice_score_vocalis2 += (2 * (vocalisPred * vocalisMask).sum()) / ((vocalisPred + vocalisMask).sum() + 1e-8)
                dice_score_laserdots3 += (2 * (laserPred * laserMask).sum()) / ((laserPred + laserMask).sum() + 1e-8)

                zwiSave = refine(laserMask, laserPred)
                if hit_chance_laser == -1.0:
                    hit_chance_laser = zwiSave
                else:
                    hit_chance_laser = (zwiSave + hit_chance_laser)/2

            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")

            print(f"Dice score for the glottis segmentation: {dice_score_glottis1/(len(loader))}")
            print(f"Dice score for the vocalis segmentation: {dice_score_vocalis2/(len(loader))}")
            print(f"Dice score for the laserdots segmentation: {dice_score_laserdots3/(len(loader))}")
            print(f"Laserdots Hit Chance: {hit_chance_laser}")
    print("#######################")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cpu"):
    print("#######################")
    print("initiate saving_img")
    if BINARY:
        model.eval()
        for idx, (x,y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                    preds = torch.sigmoid(model(x))
                    preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(y, f"{folder}/y_{idx}.png") #y.unsqueeze(1)
        
        model.train()
    else:
        model.eval()
        preds = torch.zeros(4,4,512,256)
        for idx, (x,y) in enumerate(loader):
            if idx%5 == 0:
                    print("Working on pic "+ str(idx) +" / " +str(len(loader)))
            x = x.to(device=device)
            with torch.no_grad():
                if idx >= len(loader)-1:
                    preds = torch.zeros(len(x),4,512,256)
                preds = preds.to(device)
                x = torch.cat((x,preds), 1)
                preds = torch.softmax(model(x),1)
                new_preds = refiningSoftmax(preds, True)
                new_y = []
                for i in range(0, len(y)):
                    yy = afterColor(y[i])
                    new_y.append(yy)
                new_y = np.array(new_y)
                new_y = torch.from_numpy(new_y)

            torchvision.utils.save_image(new_preds, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(new_y, f"{folder}/y_{idx}.png") #y.unsqueeze(1)
        
        model.train()
        print("#######################")
