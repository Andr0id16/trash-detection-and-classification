import cv2
import os
dir_path = r"Dataset\Aggregate\e-waste"
for image_name in os.listdir(dir_path):
    image_path = os.path.join(dir_path, image_name)
    img = cv2.imread(image_path)
    try:
        px = img[1:1]
    except Exception as e:
        print("Error")
        os.remove(image_path)
