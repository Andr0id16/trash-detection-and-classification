import json
import random
from typing import List
from PIL import Image
import os
import PIL.ImageDraw as ImageDraw
import numpy as np
import time
import math
import threading
from progress.bar import IncrementalBar


class TacoImage:
    def __init__(self, annotation):
        global images_data
        self.segmentation = annotation["segmentation"][0]
        self.points = getpoints(self.segmentation)
        self.id = annotation["image_id"]
        self.path = images_data[self.id]["file_name"]
        self.image = Image.open("data/" + self.path)


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
    center = np.average(points, axis=0)
    # mask.show()
    return mask, percentage, center


def resize(object, percentage: int):
    """uses exponential function to scale images inversely proportional to their original size"""
    scale = 0
    if percentage >= 1:
        scale = math.pow(math.e, -math.log10(4 * percentage))
    elif percentage <= 1:
        scale = math.pow(math.e, -math.log10(0.8 * percentage))
        scale = 10

    if scale < 1:
        object = object.resize(
            (int(object.size[0] * scale), int(object.size[1] * scale))
        )
    else:
        object = object.resize(
            (int(object.size[0] * scale), int(object.size[1] * scale)), Image.BICUBIC
        )

        # print(f"percentage= {percentage}% scale={scale}", flush=True)

    return object


# def transplant_segment(anno, target, paste_pos: tuple[int, int])
def transplant_segments(sample: List, target: Image):
    for no in sample:
        object = segments[no]
        target = transplant_segment(
            object, target, (random.randint(250, 450), random.randint(250, 550))
        )
    target.save(f"mt/{time.process_time_ns()+random.randint(0,10)}.png")
    bar2.next()


def transplant_segment(object: Image, target: Image, paste_pos: tuple[int, int]):

    global image_no
    # image = tacoimage.image
    # points = getpoints(tacoimage.segmentation)
    # mask, percentage, centre = getmask(image.size, points)
    # if percentage >= 1:

    #     object = Image.composite(image, mask, mask)
    #     # object.show()
    #     object = resize(object, percentage)
    target.paste(object, paste_pos, object)
    image_no += 1

    return target


def getsegments(tacoimage: TacoImage):

    global image_no, segments
    image = tacoimage.image
    points = getpoints(tacoimage.segmentation)
    mask, percentage, centre = getmask(image.size, points)
    if percentage >= 1:
        object = Image.composite(image, mask, mask)
        object = resize(object, percentage)
        segments.append(object)

    bar1.next()


def segment():
    threads = []
    for tacoimage in tacoimages:
        thread = threading.Thread(target=getsegments, args=(tacoimage,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def transplant(samples):

    threads = []
    for sample in samples:
        thread = threading.Thread(
            target=transplant_segments, args=(sample, random.choice(recipients).copy())
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":

    ## GLOBALS
    # cwd = os.getcwd()
    image_no = 5
    no_of_images = 200  # no of images to process
    with open("data/annotations.json", "r") as f:
        jsonfile = f.read()  # the json file as string
        annotations_file = json.loads(jsonfile)  # actual json as dictionary
        images_data = annotations_file["images"][200:500]  # get image data as list
        annotations = annotations_file["annotations"][
            200:500
        ]  # get annotation data as list
    segments = list()
    recipients = list()

    ## READ RECIPIENTS
    for recipient in os.listdir("recipients"):
        print(f"recipients\{recipient}")
        recipients.append(Image.open(f"recipients\{recipient}"))

    ## READ TACO IMAGES
    start_time = time.perf_counter()
    tacoimages = [TacoImage(annotation) for annotation in annotations[200:400]]
    total_time = time.perf_counter() - start_time
    print(f"Finished generating tacoimages in {total_time}s")

    ## SEGMENTING IMAGES
    start_time = time.perf_counter()
    bar1 = IncrementalBar("Segmenting", max=no_of_images)  # create a progress bar
    segment()
    bar1.finish()
    segmentation_time = time.perf_counter() - start_time
    print(f"Finished segmentation in {segmentation_time}s")

    ## TRANSPLANTING SEGMENTS
    samples = [
        random.sample(range(len(segments)), random.randint(1, 3))
        for i in range(no_of_images)
    ]
    print(samples, sep="\n")
    del tacoimages
    bar2 = IncrementalBar("Processing", max=len(samples))  # create a progress bar
    transplant(samples)
    bar2.finish()
    total_time = time.perf_counter() - start_time
    print(f"Finished execution in {total_time}s")
    print(f"Average time for processing: {total_time/no_of_images}s")
