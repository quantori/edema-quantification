import os

import cv2

from tools.models import LungSegmentation
from tools.utils import MorphologicalTransformations


def boundary_extraction(folder_dir, output_dir, model_name,thresh_val):
    model = LungSegmentation(
        model_dir=f'models/lung_segmentation/{model_name}',
        threshold=0.50,
        device='auto',
        raw_output=True,
    )
    for images in os.listdir(folder_dir):
        # check if the image ends with png or jpg or jpeg
        if images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg"):
            img_path = folder_dir + '/' + images
            mask_lungs = model(img_path)  # segmented masks
            cv2.imwrite(f'dataset/output/masks/mask_{images}.png', mask_lungs)
        morph = MorphologicalTransformations(
            image_file=f'dataset/output/masks/mask_{images}.png',
            level=3
        )
        binarized_mask = morph.convert_binary(thresh_val)  # CHANGE NAME
        boundary = morph.extract_boundary(binarized_mask)
        image_bound = morph.visualize_boundary(img_path, boundary)
        filename = os.path.split(images)[-1]
        cv2.imwrite(os.path.join(output_dir,filename),image_bound)


if __name__ == '__main__':
    dataset_dir = "C:/Users/Sunil/PycharmProjects/pythonProject/edema-quantification/dataset/data"
    output_dir = "C:/Users/Sunil/PycharmProjects/pythonProject/edema-quantification/dataset/output/boundary"
    model_name = "Unet++"
    threshold = 'Triangle'

    boundary_extraction(dataset_dir, output_dir, model_name, threshold)