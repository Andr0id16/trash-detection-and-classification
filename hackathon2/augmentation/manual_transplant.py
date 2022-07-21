import json
import random
from PIL import Image
import os
import PIL.ImageDraw as ImageDraw
import numpy as np
import time
import math
import threading
from progress.bar import IncrementalBar

cwd = os.getcwd()
image_no = 0
no_of_images = 200  # no of images to process
bar = IncrementalBar("Processing", max=no_of_images)  # create a progress bar


with open("data/annotations.json", "r") as f:
    jsonfile = f.read()  # the json file as string
    annotations = json.loads(jsonfile)  # actual json as dictionary
    images_dicts = annotations["images"][:no_of_images]  # get image data as list
    image_annotation_dicts = annotations["annotations"][
        :no_of_images
    ]  # get annotation data as list
recipient = Image.open(os.path.join(cwd, "recipients\\recipient2.jpg"))
recipient_size = recipient.size


def getpoints(seg: list[int]):
    """returns list of coordinates from segmentation data"""
    points = []
    for i in range(0, len(seg) - 1, 2):
        points.append((seg[i], seg[i + 1]))
    return points


def getmask(size: tuple, points: list[tuple[int, int]]):
    """
    returns black mask with white polygon and percentage not black"""
    mask = Image.new("RGBA", (size))
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=(255, 0, 0))  # create a polygon
    # find fraction of mask that is the object and use it to find percentage
    mask_array = np.asarray(mask)
    shape = mask_array.shape
    total_white = np.sum(mask_array)
    total = (shape[0] * shape[1]) * (255 * 3)
    percentage = total_white / total * 100
    return mask, percentage


def resize(object, percentage: int):
    """uses exponential function to scale images inversely proportional to their original size"""
    if percentage >= 1.5 or percentage <= 0.2:
        scale = math.pow(math.e, -math.log10(4 * percentage))
        object = object.resize(
            (int(object.size[0] * scale), int(object.size[1] * scale))
        )

    return object


def transplant():

    threads = []
    for anno in image_annotation_dicts[:no_of_images]:
        thread = threading.Thread(
            target=transplant_segment, args=(anno, recipient.copy())
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def transplant_segment(anno, target):
    # sema.acquire()
    global image_no
    # for anno in image_annotation_dicts:
    image_id = anno["image_id"]
    # print(image_id)
    # locks[image_id].acquire()

    paste_pos = (400, 500)
    ## SEGMENT METHOD
    image_data = images_dicts[image_id]
    image_path = image_data["file_name"]
    image_path = os.path.join(cwd, "data", image_path)
    image = Image.open(image_path)
    # print(image.size)
    if not (image.size == recipient_size):
        target = target.transpose(Image.ROTATE_90)
        paste_pos = (1200, 500)
    seg = anno["segmentation"][0]
    points = getpoints(seg)
    mask, percentage = getmask(image.size, points)

    object = Image.composite(image, mask, mask)
    object = resize(object, percentage)

    target.paste(object, paste_pos, object)

    image_no += 1
    target.save(f"mt/{time.process_time_ns()+random.randint(0,10)}.png")
    bar.next()


def bbox():
    ## BBOX METHOD
    # bbox = anno["bbox"]
    # print(image_id)
    # print(bbox)
    # xmin = bbox[0]
    # ymin = bbox[1]
    # width = bbox[2]
    # height = bbox[3]
    # xmax = width + xmin
    # ymax = height + ymin
    # image_data = images_dicts[image_id]
    # image_path = image_data["file_name"]
    # print(image_path)
    # image_path = os.path.join(cwd, "data", image_path)
    # image = Image.open(image_path)
    # cropped = image.crop((xmin, ymin, xmax, ymax))
    # print(cropped.size)
    # cropped.thumbnail((200, 200), Image.ANTIALIAS)
    # cropped_w, cropped_h = cropped.size

    # recipient_copy.paste(
    #     cropped,
    #     (
    #         int(max(0, cropped_w)),
    #         int(max(0, cropped_h)),
    #     ),
    # )
    # recipient_copy.save(f"{image_no}.jpg")
    # image_no += 1
    pass


if __name__ == "__main__":
    start_time = time.perf_counter()
    transplant()
    total_time = time.perf_counter() - start_time
    bar.finish()
    print(f"Finished execution in {total_time}s")
    print(f"Average time for processing: {total_time/no_of_images}s")
