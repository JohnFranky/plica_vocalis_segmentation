import cv2
import os
import shutil
import numpy as np

BATCH_SIZE = 4

def renameImages(images):
    new_images = []
    for name in images:
        ending = name[7:11]
        letters = name[5:7]
        numbers = name[:-6]
        new_string = letters + numbers + ending
        new_images.append(new_string)
    """
    real_length = (len(new_images)/BATCH_SIZE - int(len(new_images)/BATCH_SIZE))*4
    if real_length > 0:
        del new_images[-real_length:]
    """
    return new_images

def get_path(m):
    if(m == 0):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CF"
        anzFrames = 329
    elif(m == 1):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CM"
        anzFrames = 311
    elif(m == 2):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\DD"
        anzFrames = 350
    elif(m == 4):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\LS"
        anzFrames = 350
    elif(m == 5):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MK"
        anzFrames = 474
    elif(m == 6):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MS"
        anzFrames = 251
    elif(m == 7):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\RH"
        anzFrames = 327
    elif(m == 8):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\SS"
        anzFrames = 225
    elif(m == 9):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\TM"
        anzFrames = 603
    else:
        path = "error"
        anzFrames = -1
    return (path, anzFrames)


if __name__ == "__main__":
    all_datasets_in_validation = False
    usage_of_fh = False
    opticalflow_type = "SF\Raw"#"DUALTVL1\Raw"#
    goal_train = [2, 1, 8, 4, 0, 7, 9]
    goal_val = [5, 6]

    if len(goal_val) + len(goal_train) > 9:
        print("CHECK DATASET ORDERS")
    folder_goal = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\new_dataset"
    target = os.path.join(folder_goal, "data")
    
    try:
        shutil.rmtree(target)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    
    os.mkdir(target)
    os.mkdir(os.path.join(target, r"train_images"))
    os.mkdir(os.path.join(target, r"train_masks"))
    os.mkdir(os.path.join(target, r"train_opticalflow"))
    os.mkdir(os.path.join(target, r"val_images"))
    os.mkdir(os.path.join(target, r"val_masks"))
    os.mkdir(os.path.join(target, r"val_opticalflow"))

    os.mkdir(os.path.join(target, "val_masks", "all_4"))
    os.mkdir(os.path.join(target, "train_masks", "all_4"))

    if(usage_of_fh):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\FH"
        anzFrames = 100
    if all_datasets_in_validation:
        m = 0
        for m in range(0, 10):
            if m == 3:
                continue
            kk = get_path(m)
            anzFrames = kk[1]
            path = kk[0]
            
            trennpkt = int(anzFrames * 0.9)
            images = os.listdir(os.path.join(path, "png"))
            new_images = renameImages(images)
            masks = os.listdir(os.path.join(path, "Segmentation", "all_masks"))
            new_masks = renameImages(masks)
            for pic in range(0, trennpkt + 1):
                original = cv2.imread(os.path.join(path, "png", images[pic]), 0)
                mask = cv2.imread(os.path.join(path, "Segmentation", "all_masks", masks[pic]),0)
                cv2.imwrite(os.path.join(target, "train_images", new_images[pic]), original)
                cv2.imwrite(os.path.join(target, "train_masks", "all_4", new_masks[pic]), mask)
            
            for pic in (trennpkt + 1, anzFrames):
                original = cv2.imread(os.path.join(path, "png", images[pic]), 0)
                mask = cv2.imread(os.path.join(path, "Segmentation", "all_masks", masks[pic]),0)
                cv2.imwrite(os.path.join(target, "val_images", new_images[pic]), original)
                cv2.imwrite(os.path.join(target, "val_masks", "all_4", new_masks[pic]), mask)
    else:
        for n in range(0, len(goal_train)):
            m = goal_train[n]

            if m == 3:
                continue
            kk = get_path(m)
            anzFrames = kk[1]
            path = kk[0]

            trennpkt = int(anzFrames * 0.9)
            images = os.listdir(os.path.join(path, "png"))
            new_images = renameImages(images)
            masks = os.listdir(os.path.join(path, "Segmentation", "all_masks"))
            new_masks = renameImages(masks)

            for pic in range(0, anzFrames-1):
                original = cv2.imread(os.path.join(path, "png", images[pic]), 0)
                mask = cv2.imread(os.path.join(path, "Segmentation", "all_masks", masks[pic]),0)
                cv2.imwrite(os.path.join(target, "train_images", new_images[pic]), original)
                cv2.imwrite(os.path.join(target, "train_masks", "all_4", new_masks[pic]), mask)
                if pic > 0 and pic < anzFrames-2:
                    optflow = cv2.imread(os.path.join(path, "Segmentation", "opticalflow", opticalflow_type, images[pic]),cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(target, "train_opticalflow", new_images[pic]), optflow)
        
        for n in range(0, len(goal_val)):
            m = goal_val[n]

            if m == 3:
                continue
            kk = get_path(m)
            anzFrames = kk[1]
            path = kk[0]

            trennpkt = int(anzFrames * 0.9)
            images = os.listdir(os.path.join(path, "png"))
            new_images = renameImages(images)
            masks = os.listdir(os.path.join(path, "Segmentation", "all_masks"))
            new_masks = renameImages(masks)

            for pic in range(0, anzFrames-1):
                original = cv2.imread(os.path.join(path, "png", images[pic]), 0)
                mask = cv2.imread(os.path.join(path, "Segmentation", "all_masks", masks[pic]),0)
                cv2.imwrite(os.path.join(target, "val_images", new_images[pic]), original)
                cv2.imwrite(os.path.join(target, "val_masks", "all_4", new_masks[pic]), mask)
                if pic > 0 and pic < anzFrames-2:
                    optflow = cv2.imread(os.path.join(path, "Segmentation", "opticalflow", opticalflow_type, images[pic]),cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(target, "val_opticalflow", new_images[pic]), optflow)
        

        images = os.listdir(os.path.join(target, "train_images"))
    optflow = os.listdir(os.path.join(target, "train_opticalflow"))
    for name in images:
        if not(name in optflow):
            cv2.imwrite(os.path.join(target, "train_opticalflow", name), np.zeros_like(cv2.imread(os.path.join(target, "train_images", name), cv2.IMREAD_COLOR)))
    images = os.listdir(os.path.join(target, "val_images"))
    optflow = os.listdir(os.path.join(target, "val_opticalflow"))
    for name in images:
        if not(name in optflow):
            cv2.imwrite(os.path.join(target, "val_opticalflow", name), np.zeros_like(cv2.imread(os.path.join(target, "val_images", name), cv2.IMREAD_COLOR)))
        


        

