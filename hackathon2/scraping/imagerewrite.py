import os
import cv2 as cv

dirpath = "../augmentation/data"

print(os.listdir(dirpath))
for folder in os.listdir(dirpath):

    folderpath = os.path.join(dirpath, folder)
    # imageno = 1
    if os.path.isdir(folderpath):
        for image in os.listdir(folderpath):

            imagepath = os.path.join(folderpath, image)
            print(imagepath)
            filename, file_extension = os.path.splitext(imagepath)
            img = cv.imread(imagepath)
            newimagepath = f"{folderpath}_{filename[-6:]}{file_extension}"
            print(newimagepath)
            cv.imwrite(newimagepath, img)
            # imageno += 1
