import warnings
from typing import List, Tuple, Union

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.utils import filter_img, find_obj_bbox
from tools.supervisely_tools import convert_ann_to_mask


class SegmentationDataset(Dataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""

    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 input_size: Union[int, List[int]] = (512, 512),
                 augmentation_params=None,
                 transform_params=None) -> None:
        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.augmentation_params = augmentation_params
        self.transform_params = transform_params

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:

        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = convert_ann_to_mask(ann_path=ann_path)

        _label = torch.tensor((np.sum(mask) > 0).astype(np.int32), dtype=torch.int32)
        label = torch.unsqueeze(_label, -1).to(torch.float32)

        # Apply augmentation
        if self.augmentation_params:
            sample = self.augmentation_params(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply transformation
        if self.transform_params:
            torch_version = torch.__version__
            interpolation_image = Image.BICUBIC if torch_version <= '1.7.1' else transforms.InterpolationMode.BICUBIC
            interpolation_mask = Image.NEAREST if torch_version <= '1.7.1' else transforms.InterpolationMode.NEAREST
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     interpolation=interpolation_image),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])])
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    interpolation=interpolation_mask)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask, label


class LungsCropper(Dataset):
    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 lungs_segmentation_model=None,
                 input_size: Union[int, Tuple[int, int]] = (512, 512),
                 output_size: Union[int, Tuple[int, int]] = (512, 512),
                 transform_params=None,
                 crop_type: str = None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert crop_type in ['single_crop', 'double_crop', 'crop'], 'Invalid crop type!'
        self.crop_type = crop_type

        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.transform_params = transform_params
        self.lungs_segmentation_model = lungs_segmentation_model.to(self.device)
        self.lungs_segmentation_model = self.lungs_segmentation_model.eval()

        try:
            mean = self.transform_params['mean']
            std = self.transform_params['std']
        except Exception:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]

        self.preprocess_input_image = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.input_size,
                                                                            interpolation=Image.BICUBIC),
                                                          transforms.Normalize(mean=mean,
                                                                               std=std)])
        self.preprocess_output_image = transforms.Compose([transforms.ToTensor(),
                                                           transforms.Resize(size=self.output_size,
                                                                             interpolation=Image.BICUBIC),
                                                           transforms.Normalize(mean=mean,
                                                                                std=std)])
        self.preprocess_output_mask = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.output_size,
                                                                            interpolation=Image.NEAREST)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.ndarray],
                                             Tuple[torch.Tensor, torch.Tensor],
                                             Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)

        mask = convert_ann_to_mask(ann_path=ann_path)
        mask = cv2.resize(mask, self.input_size, self.input_size)

        if self.transform_params:
            transformed_image = self.preprocess_input_image(image).to(self.device)

            with torch.no_grad():
                lungs_prediction = self.lungs_segmentation_model(torch.unsqueeze(transformed_image, 0))
                predicted_mask = lungs_prediction.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5
            predicted_mask = filter_img(predicted_mask, contour_area=5000)
            predicted_mask = np.expand_dims(predicted_mask, 2)

            mask_intersection = mask * predicted_mask[:, :, 0]
            image_intersection = image * predicted_mask

            if self.crop_type == 'crop':
                image = self.preprocess_output_image(image_intersection)
                mask = self.preprocess_output_mask(mask_intersection)
                return image, mask

            elif self.crop_type == 'single_crop':
                bbox_coordinates = find_obj_bbox(predicted_mask)
                if len(bbox_coordinates) > 2:
                    warnings.warn('There are {} objects that might cause problems'.format(len(bbox_coordinates)))

                x_min = np.min([x[0] for x in bbox_coordinates])
                y_min = np.min([x[1] for x in bbox_coordinates])
                x_max = np.max([x[2] for x in bbox_coordinates])
                y_max = np.max([x[3] for x in bbox_coordinates])

                single_cropped_image = image_intersection[y_min:y_max, x_min:x_max]
                single_cropped_mask = mask_intersection[y_min:y_max, x_min:x_max]

                image = self.preprocess_output_image(single_cropped_image)
                mask = self.preprocess_output_mask(single_cropped_mask)
                return image, mask

            elif self.crop_type == 'double_crop':
                bbox_coordinates = find_obj_bbox(predicted_mask)
                if len(bbox_coordinates) > 2:
                    warnings.warn('There are {} objects that might cause problems'.format(len(bbox_coordinates)))

                bbox_coordinates.sort(key=lambda x: - (x[2]-x[0]) * (x[3]-x[1]))
                images = []
                masks = []

                for i, bbox in enumerate(bbox_coordinates):
                    if i >= 2:
                        break

                    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                    single_cropped_image = image_intersection[y_min:y_max, x_min:x_max]
                    single_cropped_mask = mask_intersection[y_min:y_max, x_min:x_max]

                    image = self.preprocess_output_image(single_cropped_image)
                    mask = self.preprocess_output_mask(single_cropped_mask)
                    images.append(image)
                    masks.append(mask)

                return images, masks

        return image, mask


class InferenceDataset(Dataset):
    def __init__(self,
                 img_paths: List[str],
                 input_size: Union[int, List[int]] = (512, 512)):
        self.img_paths = img_paths
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_size)
        return img, img_path


if __name__ == '__main__':

    # Example of usage of SegmentationDataset
    from tools.supervisely_tools import read_supervisely_project

    image_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir='dataset/lung_segmentation',
                                                                     included_datasets=[
                                                                         'MIMIC',
                                                                         'Vin-Dxr',
                                                                     ])
    dataset = SegmentationDataset(img_paths=image_paths,
                                  ann_paths=ann_paths,
                                  input_size=[512, 512],
                                  augmentation_params=None,
                                  transform_params=None)

    for idx in range(30):
        img, mask, label = dataset[idx]

    # Example of usage of LungsCropper
    import segmentation_models_pytorch as smp
    weights_path = 'models_lungs/PAN_se_resnet50_imagenet_210621_0826/best_weights.pth'

    model = smp.PAN(encoder_name='se_resnet50',
                    encoder_weights='imagenet',
                    activation='sigmoid',
                    in_channels=3,
                    classes=1)

    model.load_state_dict(torch.load(weights_path))
    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name='se_resnet50',
                                                                 pretrained='imagenet')
    cropper = LungsCropper(img_paths=image_paths,
                           ann_paths=ann_paths,
                           lungs_segmentation_model=model,
                           input_size=(512, 512),
                           output_size=(512, 512),
                           transform_params=preprocessing_params,
                           crop_type='single_crop')

    for idx in range(10):
        img, mask = cropper[idx]
