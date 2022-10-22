from importlib.resources import path
import os

from PIL import Image

path_in = "C:/Users/makov/Desktop/DS1/img/"
path_out = "C:/Users/makov/Desktop/edema_dataset_prototypes/"

if not os.path.exists(path_out):
    os.mkdir(path_out)

images = os.listdir(path_in)
for image in images:
    _image = os.path.splitext(image)[0]
    list_img = _image.split("_")
    print(list_img)

    left_1, top_1, right_1, bottom_1 = 0, 0, int(list_img[2]), int(list_img[2])
    area_1 = (left_1, top_1, right_1, bottom_1)
    left_2, top_2, right_2, bottom_2 = (
        int(list_img[2]),
        0,
        int(list_img[2]) + int(list_img[3]),
        int(list_img[3]),
    )
    area_2 = (left_2, top_2, right_2, bottom_2)

    img = Image.open(path_in + image)
    img_1 = img.crop(area_1)
    img_2 = img.crop(area_2)
    img_1.save(
        path_out + list_img[0] + "_" + list_img[1] + "_" + list_img[2] + "_" + "frontal" + ".png"
    )
    img_2.save(
        path_out + list_img[0] + "_" + list_img[1] + "_" + list_img[3] + "_" + "lateral" + ".png"
    )
