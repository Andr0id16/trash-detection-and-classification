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
image_no = 5
no_of_images = 30  # no of images to process
bar = IncrementalBar("Processing", max=no_of_images)  # create a progress bar

with open("data/annotations.json", "r") as f:
    jsonfile = f.read()  # the json file as string
    annotations_file = json.loads(jsonfile)  # actual json as dictionary
    images_data = annotations_file["images"][:no_of_images]  # get image data as list
    annotations = annotations_file["annotations"][
        :no_of_images
    ]  # get annotation data as list


class TacoImage:
    def __init__(self, annotation):
        global images_data
        self.segmentation = annotation["segmentation"][0]
        self.points = getpoints(self.segmentation)
        self.id = annotation["image_id"]
        self.path = images_data[self.id]["file_name"]
        self.image = Image.open("data/" + self.path)


recipients = list()
for recipient in os.listdir("recipients"):
    print(f"recipients\{recipient}")
    recipients.append(Image.open(f"recipients\{recipient}"))


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
def transplant_segment(tacoimage: TacoImage, target, paste_pos: tuple[int, int]):

    global image_no
    image = tacoimage.image
    # if not (image.size == target.size):
    #     target = target.transpose(Image.ROTATE_90)
    #     paste_pos = (1200, 500)
    points = getpoints(tacoimage.segmentation)
    mask, percentage, centre = getmask(image.size, points)
    if percentage >= 1:

        object = Image.composite(image, mask, mask)
        # object.show()
        object = resize(object, percentage)
        target.paste(object, paste_pos, object)
        # target.show()
        image_no += 1
        target.save(
            f"mt/{percentage}%_{time.process_time_ns()+random.randint(0,10)}.png"
        )
    bar.next()


start_time = time.perf_counter()
tacoimages = [TacoImage(annotation) for annotation in annotations[:1500]]
total_time = time.perf_counter() - start_time
print(f"Finished generating tacoimages in {total_time}s")
positions = [(), (), (), ()]


def transplant():

    threads = []
    for tacoimage in tacoimages:
        thread = threading.Thread(
            target=transplant_segment,
            args=(
                tacoimage,
                random.choice(recipients).copy(),
                (random.randint(250, 450), random.randint(250, 550)),
            ),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    start_time = time.perf_counter()
    transplant()
    total_time = time.perf_counter() - start_time
    bar.finish()
    print(f"Finished execution in {total_time}s")
    print(f"Average time for processing: {total_time/no_of_images}s")
