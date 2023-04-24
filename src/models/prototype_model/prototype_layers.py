from typing import List, Optional, Sequence, Dict, Union, Tuple, TypeVar
import math

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
from tqdm.auto import tqdm

from loggers import IPrototypeLogger
from utils import copy_tensor_to_nparray

T = TypeVar('T')


class IPrototypeLayer(nn.Parameter):
    """Abstract base class for edema protoype layers."""

    def update(self) -> None:
        """Exchanges the prototypes with the embeddings of real images with the closest distance"""

    def warm(self) -> None:
        """Sets grad policy for the warm training stage."""
        raise NotImplementedError

    def joint(self) -> None:
        """Sets grad policy for the joint training stage."""
        raise NotImplementedError

    def last(self) -> None:
        """Sets grad policy for the last training stage."""
        raise NotImplementedError


class PrototypeLayer(IPrototypeLayer):
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        prototype_shape: Sequence[int],
        prototype_layer_stride: int = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._shape = prototype_shape
        self._layer_stride = prototype_layer_stride
        self._epsilon = epsilon
        self._global_min_proto_dists: Optional[np.ndarray] = None
        self._global_min_fmap_patches: Optional[np.ndarray] = None

        # Dicts for storing the receptive-field and bound boxes of the prototypes. They are
        # supposed to have the following structure:
        #     0: image index in the entire dataset
        #     1: height start index
        #     2: height end index
        #     3: width start index
        #     4: width end index
        #     5: class identities
        self._proto_rf_boxes: Optional[Dict[int, Dict[str, Union[int, Sequence[int]]]]] = None
        self._proto_bound_boxes: Optional[Dict[int, Dict[str, Union[int, Sequence[int]]]]] = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_prototypes(self) -> int:
        return self._num_prototypes

    @property
    def num_prototypes_per_class(self) -> int:
        return self.num_prototypes // self.num_classes

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def layer_stride(self) -> int:
        return self._layer_stride

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def prototype_class_identity(self) -> torch.Tensor:
        # onehot indication matrix for prototypes (num_prototypes, num_classes)
        prototype_class_identity = torch.zeros(
            self.num_prototypes,
            self.num_classes,
            dtype=torch.float,
        ).cuda()
        # fills with 1 only those prototypes, which correspond to the correct class. The rest is
        # filled with 0
        for i in range(self.num_prototypes):
            prototype_class_identity[i, i // self.num_prototypes_per_class] = 1
        return prototype_class_identity

    def _create_global_min_proto_dist(self) -> np.ndarray:
        # returns ndarray for global per epoch min distances initialized by infs
        return np.full(self.num_prototypes, np.inf)

    def _create_global_min_fmap_patches(self) -> np.ndarray:
        # returns ndarray for global per epoch feature maps initialized by zeros
        return np.zeros(
            (self.num_prototypes, self.shape[1], self.shape[2], self.shape[3]),
        )

    def update(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
        logger: Optional[IPrototypeLogger] = None,
    ) -> None:
        self._global_min_proto_dists = self._create_global_min_proto_dist()
        self._global_min_fmap_patches = self._create_global_min_fmap_patches()
        self._proto_rf_boxes = {}
        self._proto_bound_boxes = {}
        with tqdm(total=len(dataloader), desc='Updating prototypes', position=3, leave=False) as t:
            for iter, batch in enumerate(dataloader):
                batch_index = _get_batch_index(iter, dataloader.batch_size)
                self._update_prototypes_on_batch(model, batch, batch_index, logger)
        self._copy_to_protoytpe_layer()
        if logger is not None:
            logger.save_boxes(self._proto_rf_boxes, 'rf', model.current_epoch)
            logger.save_boxes(self._proto_bound_boxes, 'bound', model.current_epoch)

    def _update_prototypes_on_batch(
        self,
        model: pl.LightningModule,
        batch: torch.Tensor,
        batch_index: int,
        logger: Optional[IPrototypeLogger] = None,
    ):
        images, masks, labels = _split_batch(batch)
        proto_layer_input, proto_distances = self._get_input_output_of_proto_layer(model, images)
        class_to_img_index_dict = _form_class_to_img_index_dict(self.num_classes, labels)

        for prototype_idx in range(self.num_prototypes):

            one_proto_dists = self._get_one_prototype_distances(
                prototype_idx, class_to_img_index_dict, proto_distances
            )
            if one_proto_dists is None:
                continue

            batch_min_proto_dist = np.amin(one_proto_dists)
            if batch_min_proto_dist < self._global_min_proto_dists[prototype_idx]:

                batch_argmin_proto_dist = PrototypeLayer._get_batch_argmin_proto_dist(
                    one_proto_dists
                )
                batch_argmin_proto_dist_indexed = self._change_to_batch_index(
                    prototype_idx, batch_argmin_proto_dist, class_to_img_index_dict
                )

                batch_min_fmap_patch = self._get_fmap_patch(
                    batch_argmin_proto_dist_indexed, proto_layer_input
                )

                self._global_min_proto_dists[prototype_idx] = batch_min_proto_dist
                self._global_min_fmap_patches[prototype_idx] = batch_min_fmap_patch

                rf_prototype = self._compute_rf_prototype(
                    images.size(2),
                    batch_argmin_proto_dist_indexed,
                    self._compute_proto_layer_rf_info(images.size(2), model.encoder.conv_info()),
                )

                original_img_for_shortest_proto_dist = _get_img(images, rf_prototype[0])
                masks_for_shortest_proto_dist = _get_masks(
                    masks, original_img_for_shortest_proto_dist[0]
                )

                img_crop_of_proto_rf = _crop_out_rf(
                    original_img_for_shortest_proto_dist, rf_prototype
                )

                self._save_info_in_proto_rf_boxes(prototype_idx, rf_prototype, labels, batch_index)

                prototype_distances = PrototypeLayer._get_prototype_distances(
                    proto_distances, batch_index, prototype_idx
                )
                prototype_distances_act = self._activate(prototype_distances)
                upsampled_act_distances = _upsample(
                    prototype_distances_act, original_img_for_shortest_proto_dist.shape[-1]
                )

                high_proto_activation_roi_coords = _find_high_activation_roi(
                    upsampled_act_distances
                )
                high_proto_activation_roi_crop = _get_roi_crop(
                    original_img_for_shortest_proto_dist, high_proto_activation_roi_coords
                )

                self._save_info_in_proto_bound_boxes(
                    prototype_idx, high_proto_activation_roi_coords
                )

                if logger is not None:
                    logger.save_prototype_distances(
                        proto_distances, prototype_idx, model.current_epoch
                    )
                    logger.save_graphics(
                        prototype_idx,
                        model.current_epoch,
                        rf_proto=img_crop_of_proto_rf,
                        act_roi=high_proto_activation_roi_crop,
                        original_img=original_img_for_shortest_proto_dist,
                        upsampled_act_distances=upsampled_act_distances,
                        masks=masks_for_shortest_proto_dist,
                    )

    def _get_input_output_of_proto_layer(
        self, model: pl.LightningModule, images: torch.Tensor
    ) -> np.ndarray:
        if model.training:
            model.eval()
        with torch.no_grad():
            search_batch = images.cuda()
            # this computation currently is not parallelized
            proto_layer_input_torch, proto_distances_torch = model.update_prototypes_forward(
                search_batch
            )
        proto_layer_input = copy_tensor_to_nparray(proto_layer_input_torch)
        proto_distances = copy_tensor_to_nparray(proto_distances_torch)
        del proto_layer_input_torch, proto_distances_torch
        return proto_layer_input, proto_distances

    def _get_one_prototype_distances(
        self,
        prototype_idx: int,
        class_to_img_index_dict: Dict[int, Sequence[int]],
        proto_distances: np.ndarray,
    ) -> Optional[np.ndarray]:
        # If there is not images of the target_class from this batch we go on to the next prototype
        if len(class_to_img_index_dict[self._get_target_class(prototype_idx)]) == 0:
            return None
        one_proto_dists = proto_distances[
            class_to_img_index_dict[self._get_target_class(prototype_idx)]
        ][:, prototype_idx, :, :]
        return one_proto_dists

    def _get_target_class(self, prototype_idx: int) -> int:
        # target_class is the class of the class_specific prototype
        return torch.argmax(self.prototype_class_identity[prototype_idx]).item()

    @staticmethod
    def _get_batch_argmin_proto_dist(one_proto_dists: np.ndarray) -> List[int]:
        # find arguments of the smallest distance in a matrix shape
        arg_min_flat = np.argmin(one_proto_dists)
        arg_min_matrix = np.unravel_index(arg_min_flat, one_proto_dists.shape)
        batch_argmin_proto_dist = list(arg_min_matrix)
        return batch_argmin_proto_dist

    def _change_to_batch_index(
        self,
        prototype_idx: int,
        batch_argmin_proto_dist: Sequence[int],
        class_to_img_index_dict: Dict[int, Sequence[int]],
    ) -> List[int]:
        # change the index of the smallest distance from the class specific index to the whole
        # search batch index
        batch_argmin_proto_dist[0] = class_to_img_index_dict[self._get_target_class(prototype_idx)][
            batch_argmin_proto_dist[0]
        ]
        return batch_argmin_proto_dist

    def _get_fmap_patch(
        self, batch_argmin_proto_dist_indexed: Sequence[int], proto_layer_input: np.ndarray
    ) -> np.ndarray:
        # retrieve the corresponding feature map patch
        img_index_in_batch = batch_argmin_proto_dist_indexed[0]
        fmap_height_start_index = batch_argmin_proto_dist_indexed[1] * self.layer_stride
        fmap_height_end_index = fmap_height_start_index + self.shape[2]
        fmap_width_start_index = batch_argmin_proto_dist_indexed[2] * self.layer_stride
        fmap_width_end_index = fmap_width_start_index + self.shape[3]
        batch_min_fmap_patch = proto_layer_input[
            img_index_in_batch,
            :,
            fmap_height_start_index:fmap_height_end_index,
            fmap_width_start_index:fmap_width_end_index,
        ]
        return batch_min_fmap_patch

    def _compute_proto_layer_rf_info(
        self, img_size: int, conv_info: Dict[str, int]
    ) -> List[Union[int, float]]:
        # Receptive-field information that is needed to cut out the chosen upsampled fmap patch
        _check_dimensions(conv_info)
        # Receptive field parameters for the first layer (image itself)
        rf_info = [img_size, 1, 1, 0.5]
        rf_info = _extract_network_rf_info(conv_info, rf_info)
        proto_layer_rf_info = _compute_layer_rf_info(
            layer_filter_size=self.shape[2],
            layer_stride=self.layer_stride,
            layer_padding='VALID',
            previous_layer_rf_info=rf_info,
        )
        return proto_layer_rf_info

    def _compute_rf_prototype(
        self,
        img_size: int,
        prototype_patch_index: Sequence[int],
        protoL_rf_info: List[Union[int, float]],
    ) -> List[int]:
        img_index = prototype_patch_index[0]
        rf_indices = self._compute_rf_proto_layer_at_spatial_location(
            img_size=img_size,
            height_index=prototype_patch_index[1],
            width_index=prototype_patch_index[2],
            protoL_rf_info=protoL_rf_info,
        )
        return [img_index, rf_indices[0], rf_indices[1], rf_indices[2], rf_indices[3]]

    @staticmethod
    def _compute_rf_proto_layer_at_spatial_location(
        img_size: int,
        height_index: int,
        width_index: int,
        protoL_rf_info: List[Union[int, float]],
    ):
        # computes the pixel indices of the input-image patch (e.g. 224x224) that corresponds
        # to the feature-map patch with the closest distance to the current prototype
        n = protoL_rf_info[0]
        j = protoL_rf_info[1]
        r = protoL_rf_info[2]
        start = protoL_rf_info[3]
        assert height_index <= n
        assert width_index <= n

        center_h = start + (height_index * j)
        center_w = start + (width_index * j)

        rf_start_height_index = max(int(center_h - (r / 2)), 0)
        rf_end_height_index = min(int(center_h + (r / 2)), img_size)

        rf_start_width_index = max(int(center_w - (r / 2)), 0)
        rf_end_width_index = min(int(center_w + (r / 2)), img_size)

        return [
            rf_start_height_index,
            rf_end_height_index,
            rf_start_width_index,
            rf_end_width_index,
        ]

    def _save_info_in_proto_rf_boxes(
        self,
        prototype_idx: int,
        rf_prototype: Sequence[int],
        labels: torch.Tensor,
        batch_index: int,
    ) -> None:
        # save the prototype receptive field information (pixel indices in the input image)
        self._proto_rf_boxes[prototype_idx] = {}
        self._proto_rf_boxes[prototype_idx]['image_index'] = rf_prototype[0] + batch_index
        self._proto_rf_boxes[prototype_idx]['height_start_index'] = rf_prototype[1]
        self._proto_rf_boxes[prototype_idx]['height_end_index'] = rf_prototype[2]
        self._proto_rf_boxes[prototype_idx]['width_start_index'] = rf_prototype[3]
        self._proto_rf_boxes[prototype_idx]['width_end_index'] = rf_prototype[4]
        self._proto_rf_boxes[prototype_idx]['class_indentities'] = labels[rf_prototype[0]].tolist()

    @staticmethod
    def _get_prototype_distances(
        distances: np.ndarray, batch_idx: int, prototype_idx: int
    ) -> np.ndarray:
        # Get the embeddings array of prototype distances
        return distances[batch_idx, prototype_idx, :, :]

    def _activate(self, distances: np.ndarray) -> np.ndarray:
        # The activation function of the distances is log
        return np.log((distances + 1) / (distances + self.epsilon))

    def _save_info_in_proto_bound_boxes(
        self,
        prototype_idx: int,
        roi: Tuple[int, int, int, int],
        labels: torch.Tensor,
        rf_prototype: Sequence[int],
    ) -> None:
        # save the ROI (rectangular boundary of highly activated region) dfklaswuoerw
        # the activated region can be larger than the receptive field of the patch with the
        # smallest distance
        self._proto_bound_boxes[prototype_idx] = {}
        self._proto_bound_boxes[prototype_idx]['image_index'] = self._proto_rf_boxes[prototype_idx][
            'image_index'
        ]
        self._proto_bound_boxes[prototype_idx]['height_start_index'] = roi[0]
        self._proto_bound_boxes[prototype_idx]['height_end_index'] = roi[1]
        self._proto_bound_boxes[prototype_idx]['width_start_index'] = roi[2]
        self._proto_bound_boxes[prototype_idx]['width_end_index'] = roi[3]
        self._proto_bound_boxes[prototype_idx]['class_indentities'] = labels[
            rf_prototype[0]
        ].tolist()

    def _copy_to_protoytpe_layer(self):
        prototype_update = np.reshape(self._global_min_fmap_patches, tuple(self.shape))
        self.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


def _split_batch(batch: torch.Tensor) -> torch.Tensor:
    images_and_masks = batch[0]
    images = images_and_masks[:, 0:3, :, :]
    masks = images_and_masks[:, 3:, :, :]
    labels = batch[1]
    return images, masks, labels


def _get_batch_index(iter: int, batch_size: int) -> int:
    return iter * batch_size


def _form_class_to_img_index_dict(num_classes: int, labels: torch.Tensor) -> Dict[int, List[int]]:
    # form a dict with {class:[images_idxs]}
    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    for img_index, img_y in enumerate(labels):
        img_y.tolist()
        for idx, i in enumerate(img_y):
            if i:
                class_to_img_index_dict[idx].append(img_index)
    return class_to_img_index_dict


def _check_dimensions(conv_info: Dict[str, int]) -> None:
    if len(conv_info['kernel_sizes']) != len(conv_info['strides']):
        raise Exception("The number of kernels has to be equla to the number of strides")
    if len(conv_info['kernel_sizes']) != len(conv_info['paddings']):
        raise Exception("The number of kernels has to be equla to the number of paddings")


def _get_img(batch_imgs: torch.Tensor, num_img: int) -> np.ndarray:
    # Get the whole original image where the protoype has the shortest distance
    # (num_img, img_size, img_size)
    img = batch_imgs[num_img]
    img_np = img.numpy()
    return img_np


def _get_masks(batch_masks: torch.Tensor, num_masks: int) -> np.ndarray:
    # Returns masks for the image with the shoretest prototype distance in numpy array format
    masks = batch_masks[num_masks]
    masks_np = masks.numpy()
    return masks_np


def _crop_out_rf(
    original_img_for_shortest_proto_dist: torch.Tensor, rf_prototype: Sequence[int]
) -> np.ndarray:
    # crop out the prototype receptive field from the original image
    return original_img_for_shortest_proto_dist[
        :, rf_prototype[1] : rf_prototype[2], rf_prototype[3] : rf_prototype[4]
    ]


def _extract_network_rf_info(
    conv_info: Dict[str, int], rf_info: Sequence[Union[int, float]]
) -> List[Union[int, float]]:
    for i in range(len(conv_info['kernel_sizes'])):
        rf_info = _compute_layer_rf_info(
            layer_filter_size=conv_info['kernel_sizes'][i],
            layer_stride=conv_info['strides'][i],
            layer_padding=conv_info['paddings'][i],
            previous_layer_rf_info=rf_info,
        )
    return rf_info


def _compute_layer_rf_info(
    layer_filter_size: int,
    layer_stride: int,
    layer_padding: int,
    previous_layer_rf_info: Sequence[Union[int, float]],
) -> List[Union[int, float]]:
    # based on https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    n_in = previous_layer_rf_info[0]  # receptive-field input size
    j_in = previous_layer_rf_info[1]  # receptive field jump of input layer
    r_in = previous_layer_rf_info[2]  # receptive field size of input layer
    start_in = previous_layer_rf_info[3]  # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if n_in % layer_stride == 0:
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert (
            n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        )  # sanity check
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size  # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert (
            n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        )  # sanity check
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size  # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        # with math.floor n_out of the protoype layers has (1x1) pixels less then the
        # convulution transformations from the encoder
        # n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        n_out = round((n_in - layer_filter_size + pad) / layer_stride) + 1

    pL = math.floor(pad / 2)
    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1) * j_in
    start_out = start_in + ((layer_filter_size - 1) / 2 - pL) * j_in

    return [n_out, j_out, r_out, start_out]


def _find_high_activation_roi(
    act_img: np.ndarray, percentile: int = 95
) -> Tuple[int, int, int, int]:
    # Find a high activation ROI (default treshold = 95 %)
    threshold = np.percentile(act_img, percentile)
    mask = np.ones(act_img.shape)
    mask[act_img < threshold] = 0
    lower_y = _find_lower_y(mask)
    upper_y = _find_upper_y(mask)
    lower_x = _find_lower_x(mask)
    upper_x = _find_upper_x(mask)
    return (lower_y, upper_y + 1, lower_x, upper_x + 1)


def _find_lower_y(mask: np.ndarray, lower_y: int = 0) -> int:
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    return lower_y


def _find_upper_y(mask: np.ndarray, upper_y: int = 0) -> int:
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    return upper_y


def _find_lower_x(mask: np.ndarray, lower_x: int = 0) -> int:
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    return lower_x


def _find_upper_x(mask: np.ndarray, upper_x: int = 0) -> int:
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return upper_x


def _get_roi_crop(original_img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    # Crop out the ROI with high activation from the image where the distnce for j protoype turned
    # out to be the smallest the dimensions' order of original_img_j, e.g., (224, 224, 3)
    return original_img[:, roi[0] : roi[1], roi[2] : roi[3]]


def _upsample(embeddings: T, img_size: int) -> T:
    # Upsample embeddings (e.g., (14x14)->(224x224))
    return cv2.resize(embeddings, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
