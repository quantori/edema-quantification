import numpy as np
import cv2
from PIL import Image


class MorphologicalTransformations:
    def __init__(self, image_file, level):
        self.level = 3 if level < 3 else level
        self.image_file = image_file
        self.MAX_PIXEL = 255
        self.MIN_PIXEL = 0
        self.MID_PIXEL = self.MAX_PIXEL // 2
        self.kernel = np.full(shape=(level, level), fill_value=255)

    def read_this(self):
        image_src = cv2.imread(self.image_file, 0)
        return image_src

    def convert_binary(self, image_src, thresh_val):
        color_1 = self.MAX_PIXEL
        color_2 = self.MIN_PIXEL
        initial_conv = np.where((image_src <= thresh_val), image_src, color_1)
        final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
        return final_conv

    def binarize_this(self):
        image_src = self.read_this()
        image_b = self.convert_binary(image_src=image_src, thresh_val=self.MID_PIXEL)
        return image_b

    def get_flat_submatrices(self, image_src, h_reduce, w_reduce):
        image_shape = image_src.shape
        flat_submats = np.array([
            image_src[i:(i + self.level), j:(j + self.level)]
            for i in range(image_shape[0] - h_reduce) for j in range(image_shape[1] - w_reduce)
        ])
        return flat_submats

    def erode_image(self, image_src):
        orig_shape = image_src.shape
        pad_width = self.level - 2

        image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
        pimg_shape = image_pad.shape

        h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
        flat_submats = self.get_flat_submatrices(
            image_src=image_pad, h_reduce=h_reduce, w_reduce=w_reduce
        )

        image_eroded = np.array([255 if (i == self.kernel).all() else 0 for i in flat_submats])
        image_eroded = image_eroded.reshape(orig_shape)
        return image_eroded

    def extract_boundary(self, image_src):
        image_eroded = self.erode_image(image_src=image_src)
        ext_bound = image_src - image_eroded
        return ext_bound

    def visualize_boundary(self, image_src, boundary):  # boundary is the output from extract_boundary function
        alpha = 0.5
        beta = (1.0 - alpha)
        boundary = np.expand_dims(boundary, axis=-1)
        image = Image.open(image_src)
        image = image.resize((512, 512), Image.ANTIALIAS)
        lung = np.expand_dims(image, axis=-1)
        dst = cv2.addWeighted(boundary, alpha, lung, beta, 0.0, dtype=cv2.CV_64F)
        backtorgb = cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return backtorgb


