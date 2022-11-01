from importlib.resources import path
import os
from typing import Tuple, Union
import json

from PIL import Image
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch

# path_in = "C:/Users/makov/Desktop/DS1/img/"
# path_out = "C:/Users/makov/Desktop/edema_dataset_prototypes/"

# if not os.path.exists(path_out):
#     os.mkdir(path_out)

# images = os.listdir(path_in)
# for image in images:
#     _image = os.path.splitext(image)[0]
#     list_img = _image.split("_")
#     print(list_img)

#     img = Image.open(path_in + image)
#     height = img.size[1]

#     left_1, top_1, right_1, bottom_1 = 0, 0, int(list_img[2]), height
#     area_1 = (left_1, top_1, right_1, bottom_1)
#     left_2, top_2, right_2, bottom_2 = (
#         int(list_img[2]),
#         0,
#         int(list_img[2]) + int(list_img[3]),
#         height,
#     )
#     area_2 = (left_2, top_2, right_2, bottom_2)

#     img_1 = img.crop(area_1)
#     img_2 = img.crop(area_2)
#     img_1.save(
#         path_out + list_img[0] + "_" + list_img[1] + "_" + list_img[2] + "_" + "frontal" + ".png"
#     )
#     img_2.save(
#         path_out + list_img[0] + "_" + list_img[1] + "_" + list_img[3] + "_" + "lateral" + ".png"
#     )


def make_fine_annotations_masks(num_classes, names_classes, path_to_json):

    for num_class, name_class in range(num_classes), names_classes:
        pass


def rectangle_box(pt1: np.ndarray, pt2: np.ndarray, width: int) -> Tuple:
    delta = pt2 - pt1
    print(delta)
    distance = np.linalg.norm(delta)
    rect_x = 0.5 * width * delta[0] / distance
    rect_y = 0.5 * width * delta[1] / distance

    r1 = (pt1[0] - rect_x, pt1[1] + rect_y)
    r2 = (pt1[0] + rect_x, pt1[1] - rect_y)
    r3 = (pt2[0] - rect_x, pt2[1] + rect_y)
    r4 = (pt2[0] + rect_x, pt2[1] - rect_y)

    return (r1, r2, r3, r4)


if __name__ == "__main__":
    fine_annotations_masks = {
        "Cephalization": 0,
        "Kerley": 0,
        "Effusion": 0,
        "Bat": 0,
        "Infiltrate": 0,
    }
    edema_classes = {
        "Cephalization": 0,
        "Kerley": 0,
        "Effusion": 0,
        "Bat": 0,
        "Infiltrate": 0,
    }
    with open("C:/Users/makov/Desktop/DS1/ann/10000980_54935705_1664_1664.png.json") as f:
        json_file = json.load(f)
        # print(json_file.keys())
        # print(json_file["size"])

        print(json_file["objects"][0]["tags"][0]["value"])

        for key in fine_annotations_masks.keys():
            fine_annotations_masks[key] = np.ones(
                (json_file["size"]["height"], json_file["size"]["width"])
            )

        for object in json_file["objects"]:
            if object["tags"][0]["value"] == 3:
                pass
            # print(object["tags"][0]["value"])

        # img = np.ones((json_file["size"]["height"], json_file["size"]["width"]))
        # print(img.shape)

        # rec = cv2.rectangle(img, (0, 0), (300, 300), (255, 255, 255))
        # cv2.line(rec,(0,0),(511,511),(0,0,0),1)

        # cv2.imshow("any", img)

        # cv2.waitKey(0)
