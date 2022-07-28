from __future__ import annotations
from ast import Str
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
import copy
from datetime import datetime


def getimage(image_id, width, height):
    anno = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": f"{image_id}.png",
        "license": None,
        "flickr_url": None,
        "coco_url": None,
        "date_captured": None,
        "flickr_640_url": None,
    }
    return anno


def getannotation(image_id, category_id, segmentation_data, bbox, iscrowd):
    global annotation_no
    annotation_no += 1
    return {
        "id": annotation_no,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": [],
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }


class TacoImage:
    def __init__(self, annotation):
        global images_data
        self.segmentation = annotation["segmentation"][0]
        self.bbox = annotation["bbox"]
        self.id = annotation["image_id"]
        self.image = Image.open("data/" + images_data[self.id]["file_name"])
        self.cat = annotation["category_id"]
        self.iscrowd = annotation["iscrowd"]


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


def resize(object, bbox, segmentation, percentage: int):
    """uses exponential function to scale images inversely proportional to their original size"""
    scale = 0
    scale = math.pow(math.e, -math.log10(4 * percentage))
    object = object.resize((int(object.size[0] * scale), int(object.size[1] * scale)))
    bbox = list(map(lambda x: scale * x, bbox))
    segmentation = list(map(lambda x: scale * x, segmentation))
    return object, bbox, segmentation


# def transplant_segment(anno, target, paste_pos: tuple[int, int])
def transplant_segments(sample: List, target: Image):

    filename = f"{time.process_time_ns()}{random.randint(0, 10)}{threading.get_ident()}"
    filename = int(filename)
    for no in sample:
        tacoimage = segments[no]
        xmax = target.size[0] - tacoimage.bbox[0] - tacoimage.bbox[2]
        ymax = target.size[1] - tacoimage.bbox[1] - tacoimage.bbox[3]
        xmax = xmax if xmax > 0 else 1
        ymax = ymax if ymax > 0 else 1
        paste_pos = (
            random.randint(0, int(xmax)),
            random.randint(0, int(ymax)),
        )
        temp_bbox = copy.deepcopy(tacoimage.bbox)
        temp_segmentation = copy.deepcopy(tacoimage.segmentation)
        # modifying bbox and segmentation for pasted object
        temp_bbox[0:2] = [
            paste_pos[0] + tacoimage.bbox[0],
            paste_pos[1] + tacoimage.bbox[1],
        ]
        for index in range(len(tacoimage.segmentation)):
            temp_segmentation[index] += paste_pos[index % 2]

        # add new annotation
        new_annotations.append(
            getannotation(
                filename, tacoimage.cat, temp_segmentation, temp_bbox, tacoimage.iscrowd
            )
        )

        # transplant segment
        target = transplant_segment(tacoimage.image, target, paste_pos)

    target.save(f"transplant_dataset/{filename}.png")
    new_images.append(getimage(filename, *target.size))
    bar2.next()


def transplant_segment(object_image, target: Image, paste_pos: tuple[int, int]):

    target.paste(object_image, paste_pos, object_image)
    return target


def getsegments(tacoimage: TacoImage):

    global segments
    image = tacoimage.image
    points = getpoints(tacoimage.segmentation)
    mask, percentage, centre = getmask(image.size, points)
    if percentage >= 1:
        object = Image.composite(image, mask, mask)
        tacoimage.image, tacoimage.bbox, tacoimage.segmentation = resize(
            object, tacoimage.bbox, tacoimage.segmentation, percentage
        )
        segments.append(tacoimage)

    bar1.next()


def segment(start=0):
    stop = start + 100
    if start >= no_of_images:
        return
    threads = []
    for tacoimage in tacoimages[start:stop]:
        thread = threading.Thread(target=getsegments, args=(tacoimage,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    segment(stop)


def transplant(samples, start=0):
    stop = start + 100
    if start >= no_of_images:
        return
    threads = []
    for sample in samples[start:stop]:
        thread = threading.Thread(
            target=transplant_segments, args=(sample, random.choice(recipients).copy())
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    transplant(samples, stop)


if __name__ == "__main__":

    ## GLOBALS
    # cwd = os.getcwd()
    annotation_no = 0
    new_annotations = []
    new_images = []
    licenses = []
    no_of_images = 1500  # no of images to process

    with open("data/annotations.json", "r") as f:
        jsonfile = f.read()  # the json file as string
        annotations_file = json.loads(jsonfile)  # actual json as dictionary
        images_data = annotations_file["images"][
            :no_of_images
        ]  # get image data as list
        annotations = annotations_file["annotations"][
            :no_of_images
        ]  # get annotation data as list
        categories = annotations_file["categories"][:no_of_images]
    segments = list()
    recipients = list()

    ## READ RECIPIENTS
    for recipient in os.listdir("recipients"):
        print(f"recipients\{recipient}")
        recipients.append(Image.open(f"recipients\{recipient}"))

    ## READ TACO IMAGES
    start_time = time.perf_counter()
    tacoimages = [TacoImage(annotation) for annotation in annotations[:no_of_images]]
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
        for _ in range(no_of_images)
    ]
    del tacoimages
    bar2 = IncrementalBar("Processing", max=len(samples))  # create a progress bar
    transplant(samples)
    bar2.finish()

    new_annotation_file = {
        "info": {
            "year": 2022,
            "version": None,
            "description": "TACO Tranplant",
            "contributor": None,
            "url": None,
            "date_created": f"{datetime.now().strftime('%H:%M:%S')}",
        },
        "images": new_images,
        "annotations": new_annotations,
        "scene_annotations": None,
        "licenses": licenses,
        "categories": categories,
        "scene_categories": None,
    }
    with open("transplant.json", "w") as f:
        f.write(json.dumps(new_annotation_file))

    total_time = time.perf_counter() - start_time
    print(f"Finished execution in {total_time}s")
    print(f"Average time for processing: {total_time/no_of_images}s")
