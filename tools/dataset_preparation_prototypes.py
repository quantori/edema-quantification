# from importlib.resources import path
# import os
from typing import Tuple, Union
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch

# from PIL import Image

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


def rectangle_box(pt1, pt2, width) -> Tuple:

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


coords = rectangle_box(pt1=np.array([23, 45]), pt2=np.array([34, 56]), width=5)
print(coords)

# plt.xlim(0, 100)
# plt.ylim(0, 100)
# ax.axline((pt1[0], pt1[1]), (pt2[0], pt2[1]), linewidth=1, color="r")
# points = [r1, r2, r4, r3]
# rect = patches.Polygon(points, linewidth=1, edgecolor="r")
# ax.add_patch(rect)
# ax.scatter(x=r1[0], y=r1[1])
# ax.scatter(x=r2[0], y=r2[1])
# ax.scatter(x=r3[0], y=r3[1])
# ax.scatter(x=r4[0], y=r4[1])
# plt.show()
