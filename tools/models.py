import os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Union

import cv2
import wandb
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

from tools.utils import  filter_img, find_obj_bbox


class SegmentationModel:
    def __init__(self,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 aux_params: Dict = None,
                 epochs: int = 30,
                 input_size: Union[int, List[int]] = (512, 512),
                 in_channels: int = 3,
                 num_classes: int = 1,
                 class_name: str = 'Segmented Lungs',
                 save_dir: str = 'models',
                 activation:str='sigmoid',
                 wandb_api_key: str = 'b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871',
                 wandb_project_name: str = 'lung_segmentation') -> None:

        # Dataset settings
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        # Device settings
        self.device = self.device_selection()

        # Model settings
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.aux_params = aux_params
        self.epochs = epochs
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.activation = activation
        run_time = datetime.now().strftime("%d%m%y_%H%M")
        self.run_name = '{:s}_{:s}_{:s}_{:s}'.format(self.model_name, self.encoder_name, self.encoder_weights, run_time)
        self.model_dir = os.path.join(save_dir, self.run_name)

        # Logging settings
        self.wandb_api_key = wandb_api_key
        self.wandb_project_name = wandb_project_name

    def get_hyperparameters(self) -> Dict[str, Any]:
        hyperparameters = {
            'model_name': self.model_name,
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'aux_params': self.aux_params,
            'epochs': self.epochs,
            'img_height': self.input_size[0],
            'img_width': self.input_size[1],
            'img_channels': self.in_channels,
            'classes': self.num_classes,
            'activation': self.activation,
            'device': self.device
        }
        return hyperparameters

    @staticmethod
    def _get_log_params(model: Any,
                        img_height: int,
                        img_width: int,
                        img_channels: int) -> Dict[str, float]:
        from ptflops import get_model_complexity_info
        _macs, _params = get_model_complexity_info(model, (img_channels, img_height, img_width),
                                                   as_strings=False, print_per_layer_stat=False, verbose=False)
        macs = round(_macs / 10. ** 9, 1)
        params = round(_params / 10. ** 6, 1)
        params = {'params': params, 'macs': macs}
        return params

    @staticmethod
    def _get_log_metrics(train_logs: Dict[str, float],
                         val_logs: Dict[str, float],
                         test_logs: Dict[str, float],
                         prefix: str = '') -> Dict[str, float]:
        train_metrics = {prefix + 'train/' + k: v for k, v in train_logs.items()}
        val_metrics = {prefix + 'val/' + k: v for k, v in val_logs.items()}
        test_metrics = {prefix + 'test/' + k: v for k, v in test_logs.items()}
        metrics = {}
        for m in [train_metrics, val_metrics, test_metrics]:
            metrics.update(m)
        return metrics

    def _get_log_images(self,
                        model: Any,
                        log_image_size: Tuple[int, int],
                        logging_loader: torch.utils.data.dataloader.DataLoader) -> Tuple[List[wandb.Image], List[wandb.Image]]:

        model.eval()
        mean = torch.tensor(logging_loader.dataset.transform_params['mean'])
        std = torch.tensor(logging_loader.dataset.transform_params['std'])

        with torch.no_grad():
            segmentation_masks = []
            probability_maps = []
            for idx, (image, mask, label) in enumerate(logging_loader):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image, mask, label = image.to(device), mask.to(device), label.to(device)
                pred_seg = model(image)
                if isinstance(pred_seg, tuple):
                    pred_seg, pred_cls = pred_seg
                    pred_cls = torch.round(pred_cls.view(-1))
                    pred_seg = pred_seg * pred_cls.view(-1, 1, 1, 1)

                image_bg = torch.clone(image).squeeze(dim=0)
                image_bg = image_bg.permute(1, 2, 0)
                image_bg = (((image_bg.detach().cpu() * std) + mean) * 255).numpy().astype(np.uint8)
                image_bg = cv2.resize(image_bg, log_image_size, interpolation=cv2.INTER_CUBIC)

                mask_gt = torch.clone(mask).squeeze()
                mask_gt = mask_gt.detach().cpu().numpy().astype(np.uint8)
                mask_gt = cv2.resize(mask_gt, log_image_size, interpolation=cv2.INTER_NEAREST)

                mask_pred = torch.clone(pred_seg).squeeze()
                mask_pred = (mask_pred > 0.5).detach().cpu().numpy().astype(np.uint8)
                mask_pred = cv2.resize(mask_pred, log_image_size, interpolation=cv2.INTER_NEAREST)

                prob_map = torch.clone(pred_seg).squeeze()
                prob_map = (prob_map * 255).detach().cpu().numpy().astype(np.uint8)
                prob_map = cv2.resize(prob_map, log_image_size, interpolation=cv2.INTER_CUBIC)

                segmentation_masks.append(wandb.Image(image_bg,
                                                      masks={'Prediction': {'mask_data': mask_pred,
                                                                            'class_labels': self.logging_labels},
                                                             'Ground truth': {'mask_data': mask_gt,
                                                                              'class_labels': self.logging_labels},
                                                             },
                                                      caption='Mask {:d}'.format(idx + 1)))
                probability_maps.append(wandb.Image(prob_map,
                                                    masks={'Ground truth': {'mask_data': mask_gt,
                                                                            'class_labels': self.logging_labels}},
                                                    caption='Map {:d}'.format(idx + 1)))

        model.train()
        return segmentation_masks, probability_maps

    def print_model_settings(self) -> None:
        print('\033[1m\033[4m\033[93m' + '\nModel settings:' + '\033[0m')
        print('\033[92m' + 'Class name:       {:s}'.format(self.class_name) + '\033[0m')
        print('\033[92m' + 'Model name:       {:s}'.format(self.model_name) + '\033[0m')
        print('\033[92m' + 'Encoder name:     {:s}'.format(self.encoder_name) + '\033[0m')
        print('\033[92m' + 'Weights used:     {:s}'.format(self.encoder_weights) + '\033[0m')
        print('\033[92m' + 'Input size:       {:d}x{:d}x{:d}'.format(self.input_size[0],
                                                                     self.input_size[1],
                                                                     self.in_channels) + '\033[0m')
        print('\033[92m' + 'Activation:       {:s}'.format(self.activation) + '\033[0m')
        print('\033[92m' + 'Class count:      {:d}'.format(self.num_classes) + '\033[0m')
    def device_selection(self) -> str:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GPU
        n = torch.cuda.device_count()
        if n > 1 and self.batch_size:
            assert self.batch_size % n == 0, 'batch size {:d} does not multiple of GPU count {:d}'.format(
                self.batch_size, n)
        gpu_s = ''
        for idx in range(n):
            p = torch.cuda.get_device_properties(idx)
            gpu_s += "{:s}, {:.0f} MB".format(p.name, p.total_memory / 1024 ** 2)

        # CPU
        from cpuinfo import get_cpu_info
        cpu_info = get_cpu_info()
        cpu_s = "{:s}, {:d} cores".format(cpu_info['brand_raw'], cpu_info["count"])

        print('\033[1m\033[4m\033[93m' + '\nDevice settings:' + '\033[0m')
        if device == 'cuda':
            print('\033[92m' + '✅ GPU: {:s}'.format(gpu_s) + '\033[0m')
            print('\033[91m' + '❌ CPU: {:s}'.format(cpu_s) + '\033[0m')
        else:
            print('\033[92m' + '✅ CPU: {:s}'.format(cpu_s) + '\033[0m')
            print('\033[91m' + '❌ GPU: ({:s})'.format(gpu_s) + '\033[0m')
        return device

    def build_model(self) -> Any:
        if self.model_name == 'Unet':
            model = smp.Unet(encoder_name=self.encoder_name,
                             encoder_weights=self.encoder_weights,
                             in_channels=self.in_channels,
                             classes=self.num_classes,
                             activation=self.activation,
                             aux_params=self.aux_params)
        elif self.model_name == 'Unet++':
            model = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                     encoder_weights=self.encoder_weights,
                                     in_channels=self.in_channels,
                                     classes=self.num_classes,
                                     activation=self.activation,
                                     aux_params=self.aux_params)
        elif self.model_name == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=self.encoder_name,
                                  encoder_weights=self.encoder_weights,
                                  in_channels=self.in_channels,
                                  classes=self.num_classes,
                                  activation=self.activation,
                                  aux_params=self.aux_params)
        elif self.model_name == 'DeepLabV3+':
            model = smp.DeepLabV3Plus(encoder_name=self.encoder_name,
                                      encoder_weights=self.encoder_weights,
                                      in_channels=self.in_channels,
                                      classes=self.num_classes,
                                      activation=self.activation,
                                      aux_params=self.aux_params)
        elif self.model_name == 'FPN':
            model = smp.FPN(encoder_name=self.encoder_name,
                            encoder_weights=self.encoder_weights,
                            in_channels=self.in_channels,
                            classes=self.num_classes,
                            activation=self.activation,
                            aux_params=self.aux_params)
        elif self.model_name == 'Linknet':
            model = smp.Linknet(encoder_name=self.encoder_name,
                                encoder_weights=self.encoder_weights,
                                in_channels=self.in_channels,
                                classes=self.num_classes,
                                activation=self.activation,
                                aux_params=self.aux_params)
        elif self.model_name == 'PSPNet':
            model = smp.PSPNet(encoder_name=self.encoder_name,
                               encoder_weights=self.encoder_weights,
                               in_channels=self.in_channels,
                               classes=self.num_classes,
                               activation=self.activation,
                               aux_params=self.aux_params)
        elif self.model_name == 'PAN':
            model = smp.PAN(encoder_name=self.encoder_name,
                            encoder_weights=self.encoder_weights,
                            in_channels=self.in_channels,
                            classes=self.num_classes,
                            activation=self.activation,
                            aux_params=self.aux_params)
        elif self.model_name == 'MAnet':
            model = smp.MAnet(encoder_name=self.encoder_name,
                              encoder_weights=self.encoder_weights,
                              in_channels=self.in_channels,
                              classes=self.num_classes,
                              activation=self.activation,
                              aux_params=self.aux_params)
        else:
            raise ValueError('Unknown model name: {:s}'.format(self.model_name))

        return model

class LungSegmentation:
    def __init__(self,
                 lungs_segmentation_model,
                 device,
                 threshold: float,
                 lung_input_size: Union[int, List[int]] = (512, 512),
                 lung_preprocessing=None,
                 crop_type: str = None):
        assert 0 <= threshold <= 1, 'Threshold value is in an incorrect scale. It should be in the range [0,1].'
        assert crop_type in ['no_crop', 'crop', 'single_crop'], 'invalid flag type'
        self.device = device
        self.threshold = threshold

        self.lungs_segmentation = lungs_segmentation_model.to(device).eval()
        self.lung_input_size = (lung_input_size, lung_input_size) if isinstance(lung_input_size, int) else lung_input_size
        self.lung_preprocessing = lung_preprocessing
        self.crop_type = crop_type

        self.preprocess_image_lung = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=self.lung_input_size,
                                                                          interpolation=Image.BICUBIC),
                                                        transforms.Normalize(mean=self.lung_preprocessing['mean'],
                                                                             std=self.lung_preprocessing['std'])])

    def __call__(self, img):
        return self.predict(img)

    def eval(self):
        self.lungs_segmentation.eval()

    def predict_masks(self, source_img):
        lung_img = torch.unsqueeze(self.preprocess_image_lung(source_img), dim=0).to(self.device)

        if self.crop_type == 'no_crop':
            mask_lungs = self.lungs_segmentation(lung_img)[0, 0, :, :].cpu().detach().numpy()
            mask_lungs = cv2.resize(mask_lungs, (512, 512))
            return mask_lungs

        if self.crop_type == 'crop':
            mask_lungs = self.lungs_segmentation(lung_img).permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5
            mask_lungs = cv2.resize(mask_lungs[:, :, 0].astype(np.uint8), (512, 512))
            return mask_lungs

        if self.crop_type == 'single_crop':
            mask_lungs = self.lungs_segmentation(lung_img).permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5
            mask_lungs = filter_img(mask_lungs, contour_area=6000)
            mask_lungs = np.expand_dims(mask_lungs, 2)

            crop_lungs = source_img * mask_lungs
            bbox_coordinates = find_obj_bbox(mask_lungs)
            if len(bbox_coordinates) == 0:
                height, width, _ = crop_lungs.shape
                bbox_coordinates = np.array([[0, 0, width - 1, height - 1]])

            bbox_min_x = np.min([x[0] for x in bbox_coordinates])
            bbox_min_y = np.min([x[1] for x in bbox_coordinates])
            bbox_max_x = np.max([x[2] for x in bbox_coordinates])
            bbox_max_y = np.max([x[3] for x in bbox_coordinates])

            single_cropped_lungs_predicted = mask_lungs[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
            mask_lungs = cv2.resize(single_cropped_lungs_predicted.astype(np.uint8), (512, 512))
            return mask_lungs

    """def get_segment_crop(img, tol=0, mask=None):
        if mask is None:
            mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]"""

    def predict(self, source_img):
        assert source_img.shape[2] == 3, 'Incorrect image dimensions'
        pred_mask_lungs = self.predict_masks(source_img)
        return pred_mask_lungs
