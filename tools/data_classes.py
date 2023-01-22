import os
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tools.data_classes_utils as data_classes_utils


class EdemaDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        make_augmentation: bool = False,
        normalize_tensors: bool = False,
        resize: Tuple[int, int] = (500, 500),
    ) -> None:
        self.img_data = self.process_metadata_df(metadata_df)
        self.make_augmentation = make_augmentation
        self.normalize_tensors = normalize_tensors
        self.resize = resize

    @staticmethod
    def process_metadata_df(metadata_df: pd.DataFrame) -> Dict[int, Dict]:
        """

        Args:
            metadata_df: DataFrame obtained from 'metadata.xlsx' spreadsheet table
                in the folder with converted images

        Returns:
            Dict[int, Dict[str, Union[str, int, Dict]]] dict of dicts with data
                extracted from image metadata
        """
        img_data = dict()
        for img_idx, (img_path, img_objects) in enumerate(
            metadata_df.groupby('Image path', sort=False)
        ):
            img_data[img_idx] = dict()
            img_data[img_idx]['path'] = img_path
            img_data[img_idx]['label'] = int(img_objects['Class ID'].iloc[0])
            annotations = data_classes_utils.extract_annotations(img_objects)
            img_data[img_idx]['annotations'] = annotations

        return img_data

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data[idx]['path']
        image = Image.open(img_path)
        label = self.img_data[idx]['label']
        annotations = self.img_data[idx]['annotations']

        # resize image to target size and create list with masks, as well as labels array
        image_arr, masks, findings = data_classes_utils.resize_and_create_masks(
            image, annotations, self.resize
        )

        if self.make_augmentation:
            augmentation_functions = [
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.GaussianBlur(p=0.2),
            ]
        else:
            augmentation_functions = [A.NoOp(p=1.0)]

        if self.normalize_tensors:
            # these are mean/std values for RGB channels in a set of 110 annotated images (DS1, DS2 folders)
            normalization = A.Normalize(mean=[0.4675, 0.4675, 0.4675], std=[0.3039, 0.3039, 0.3039])
        else:
            normalization = A.NoOp(p=1.0)

        transform = A.Compose(
            [
                *augmentation_functions,
                normalization,
                ToTensorV2(),
            ]
        )

        transformed = transform(image=image_arr, masks=masks)

        tensor = data_classes_utils.combine_image_and_masks(
            transformed['image'], transformed['masks']
        )
        return tensor, findings


class EdemaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = './',
        batch_size: int = 32,
        resize: Tuple[int, int] = (500, 500),
        make_augmentation: bool = False,
        normalize_tensors: bool = False,
        train_share: float = 0.8,
    ) -> None:
        """
        DataModule for preparing batches of processed images.

        Args:
            data_dir: directory where converted supervisely dataset reside
            batch_size: batch size
            resize: tuple with desired input size (width, height) of the processed images
            make_augmentation: whether to apply a set of augmentation operations
            normalize_tensors: whether to normalize output tensors
            train_share: share of the train part
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.make_augmentation = make_augmentation
        self.normalize_tensors = normalize_tensors
        self.train_share = train_share

    def setup(self, stage):
        metadata_df = pd.read_excel(os.path.join(self.data_dir, 'metadata.xlsx'))
        edema_full = EdemaDataset(
            metadata_df,
            make_augmentation=self.make_augmentation,
            normalize_tensors=self.normalize_tensors,
            resize=self.resize,
        )
        if stage == 'fit':
            self.edema_train, self.edema_test = data_classes_utils.split_dataset(
                edema_full, self.train_share, metadata_df, verbose=True
            )

    def train_dataloader(self, num_workers=1):
        return DataLoader(self.edema_train, batch_size=self.batch_size, num_workers=num_workers)

    def test_dataloader(self, num_workers=1):
        return DataLoader(self.edema_test, batch_size=self.batch_size, num_workers=num_workers)


if __name__ == '__main__':

    metadata_df = pd.read_excel(
        os.path.join(
            'C:/Users/makov/Desktop/edema-quantification/dataset/MIMIC-CXR-Edema-Intermediate',
            'metadata.xlsx',
        )
    ).fillna({'Class ID': -1})
    dataset = EdemaDataset(metadata_df, normalize_tensors=False)
    images, labels = dataset[1]
    print(images.shape)

    datamodule = EdemaDataModule(
        data_dir='C:/Users/makov/Desktop/edema-quantification/dataset/MIMIC-CXR-Edema-Intermediate'
    )
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    print('train dataloader')
    for idx, batch in enumerate(train_dataloader):
        images, labels = batch
        print('batch_' + str(idx))
        print(labels)
    print('test dataloader')
    for idx, batch in enumerate(test_dataloader):
        images, labels = batch
        print('batch_' + str(idx))
        print(labels)

    # images, labels = next(iter(train_dataloader))
    # print(images.dtype, labels.dtype)

    # img_num = 31
    # masks = images[img_num, 4:5, :, :]
    # images = images[img_num, 0:3, :, :]
    # import torch

    # print(images.shape, torch.max(images), torch.min(images))
    # print(labels[img_num].dtype)
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.imshow(images.permute(1, 2, 0))
    # plt.figure()
    # plt.imshow(masks.permute(1, 2, 0))
    # plt.show()
