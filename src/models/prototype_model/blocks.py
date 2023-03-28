from typing import List, Optional, Sequence, Dict, Union, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pytorch_lightning as pl

from src.models.prototype_model.utils import _make_layers
from utils import ImageSaver, copy_tensor_to_nparray

####################################################################################################
# UPDATER
####################################################################################################

class ProtoUpdater(ABC):
    """Abstract base class for implementing an updater for prototypes."""

    @abstractmethod
    def update_prototypes():
        """Implement this at subclasses."""


class ProtoUpdaterParameter(ProtoUpdater):
    """Class for updating prototypes based on Parameter class from PyTorch.

    The class performs all the necessary internal work to update protoypes.

    Args:
        model: the pl.LightningModule subclass from which this class is called.
        dataloader: DataLoader with the data.
        proto_rf_boxes: initial dict for storing receptive field boxes of the prototypes.
        proto_bound_boxes: initial dict for storing bound boxes based on the activations of the
            prototypes.
    """

    def __init__(self, prototype_layer: nn.Parameter) -> None:
        self._prototype_layer = prototype_layer
        self._global_min_proto_dists = self._create_global_min_proto_dist()
        self._global_min_fmap_patches = self._create_global_min_fmap_patches()
        self._proto_rf_boxes = ProtoUpdaterParameter._create_proto_rf_boxes()
        self._proto_bound_boxes = ProtoUpdaterParameter._create_proto_bound_boxes()

    def update_prototypes(self, model: pl.LightningModule, dataloader: DataLoader, saver: Optional[ImageSaver]) -> None:
        with tqdm(total=len(dataloader), desc='Updating prototypes', position=3, leave=False) as t:
            for iter, batch in enumerate(dataloader):
                batch_index = _get_batch_index(iter, dataloader.batch_size)
                self._update_prototypes_on_batch(model, batch, batch_index, saver)

    def _update_prototypes_on_batch(
        self, model: pl.LightningModule, batch: torch.Tensor, batch_index: int, saver: Optional[ImageSaver]
    ):
        images, masks, labels = _split_batch(batch)
        proto_layer_input, proto_distances = ProtoUpdaterParameter._get_input_output_of_proto_layer(model, images)
        class_to_img_index_dict = _form_class_to_img_index_dict(model.num_classes, labels)

        for prototype_idx in range(self._prototype_layer.num_prototypes):
            one_proto_dists = self._get_one_prototype_distances(prototype_idx, class_to_img_index_dict, proto_distances)
            if one_proto_dists is None: continue
            batch_min_proto_dist = ProtoUpdaterParameter._get_batch_min_proto_dist(one_proto_dists)
            if batch_min_proto_dist < self._global_min_proto_dists[prototype_idx]:
                batch_argmin_proto_dist = ProtoUpdaterParameter._get_batch_argmin_proto_dist(one_proto_dists)
                batch_argmin_proto_dist_indexed = self._change_index(prototype_idx, batch_argmin_proto_dist, class_to_img_index_dict)
                batch_min_fmap_patch = self._get_fmap_patch(batch_argmin_proto_dist_indexed, proto_layer_input)
                self._set_global_min_proto_dist(prototype_idx, batch_min_proto_dist)
                self._set_global_min_fmap_patch(prototype_idx, batch_min_fmap_patch)

    def _set_global_min_proto_dist(self, prototype_idx: int, batch_min_proto_dist: float) -> None:
        self._global_min_proto_dists[prototype_idx] = batch_min_proto_dist

    def _set_global_min_fmap_patch(self, prototype_idx: int, batch_min_fmap_patch: np.ndarray) -> None:
        self._global_min_fmap_patches[prototype_idx] = batch_min_fmap_patch

    def _get_fmap_patch(self, batch_argmin_proto_dist_indexed: Sequence[int], proto_layer_input: np.ndarray) -> np.ndarray:
        # retrieve the corresponding feature map patch
        img_index_in_batch = batch_argmin_proto_dist_indexed[0]
        fmap_height_start_index = batch_argmin_proto_dist_indexed[1] * self._prototype_layer.layer_stride
        fmap_height_end_index = fmap_height_start_index + self._prototype_layer.shape[2]
        fmap_width_start_index = batch_argmin_proto_dist_indexed[2] * self._prototype_layer.layer_stride
        fmap_width_end_index = fmap_width_start_index + self._prototype_layer.shape[3]
        batch_min_fmap_patch = proto_layer_input[
            img_index_in_batch,
            :,
            fmap_height_start_index:fmap_height_end_index,
            fmap_width_start_index:fmap_width_end_index,
        ]
        return batch_min_fmap_patch

    def _get_target_class(self, prototype_idx: int) -> int:
        # target_class is the class of the class_specific prototype
        return torch.argmax(self._prototype_layer.make_prototype_class_identity()[prototype_idx]).item()

    @staticmethod
    def _get_input_output_of_proto_layer(model: pl.LightningModule, images: torch.Tensor) -> np.ndarray:
        if model.training: model.eval()
        with torch.no_grad():
            search_batch = images.cuda()
            # this computation currently is not parallelized
            proto_layer_input_torch, proto_distances_torch = model.update_prototypes_forward(search_batch)
        proto_layer_input = copy_tensor_to_nparray(proto_layer_input_torch)
        proto_distances = copy_tensor_to_nparray(proto_distances_torch)
        del proto_layer_input_torch, proto_distances_torch
        return proto_layer_input, proto_distances   

    def _get_one_prototype_distances(self, prototype_idx: int, class_to_img_index_dict: Dict[int, Sequence[int]], proto_distances: np.ndarray) -> Optional[np.ndarray]:
        # if there is not images of the target_class from this batch we go on to the next prototype
        if len(class_to_img_index_dict[self._get_target_class(prototype_idx)]) == 0:
            return None
        one_proto_dists = proto_distances[class_to_img_index_dict[self._get_target_class(prototype_idx)]][:, prototype_idx, :, :]
        return one_proto_dists
    
    def _change_index(self, prototype_idx: int, batch_argmin_proto_dist: Sequence[int], class_to_img_index_dict: Dict[int, Sequence[int]]) -> List[int]:
        # change the index of the smallest distance from the class specific index to the whole
        # search batch index
        batch_argmin_proto_dist[0] = class_to_img_index_dict[self._get_target_class(prototype_idx)][
            batch_argmin_proto_dist[0]
        ] 
        return batch_argmin_proto_dist

    @staticmethod
    def _get_batch_min_proto_dist(one_proto_dists: np.ndarray) -> float:
        return np.amin(one_proto_dists)

    @staticmethod
    def _get_batch_argmin_proto_dist(one_proto_dists: np.ndarray) -> List[int]:
        # find arguments of the smallest distance in a matrix shape
        arg_min_flat = np.argmin(one_proto_dists)
        arg_min_matrix = np.unravel_index(arg_min_flat, one_proto_dists.shape)
        batch_argmin_proto_dist = list(arg_min_matrix)
        return batch_argmin_proto_dist

    def _create_global_min_proto_dist(self) -> np.array:
        # returns tensor for global per epoch min distances initialized by infs
        return np.full(self._prototype_layer.num_prototypes, np.inf)

    def _create_global_min_fmap_patches(self) -> np.array:
        # returns tensor for global per epoch feature maps initialized by zeros
        return np.zeros(
            (self._prototype_layer.num_prototypes, self._prototype_layer.prototype_shape[1], self._prototype_layer.prototype_shape[2], self._prototype_layer.prototype_shape[3]),
        )

    @staticmethod
    def _create_proto_rf_boxes() -> Dict[int, Union[int, Sequence[int]]]:
        # initial dict for storing receptive field boxes of the prototypes. It is supposed to have
        # the following structure:
        #     0: image index in the entire dataset
        #     1: height start index
        #     2: height end index
        #     3: width start index
        #     4: width end index
        #     5: class identities
        return {}

    @staticmethod
    def _create_proto_bound_boxes() -> Dict[int, Union[int, Sequence[int]]]:
        # initial dict for storing bound boxes based on the activations of the prototypes. It is
        # suposed to have the same structure as proto_rf_boxes
        return {}


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


def update_prototypes_on_batch(
    self,
    search_batch_images,
    start_index_of_search_batch,
    search_labels,
    Callable[int, List[int]],
    global_min_proto_dist,  # this will be updated
    global_min_fmap_patches,  # this will be updated
    proto_rf_boxes,  # this will be updated
    proto_bound_boxes,  # this will be updated
    prototype_layer_stride=1,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
):
    # Model has to be in the eval mode
    if self.training:
        self.eval()

    with torch.no_grad():
        search_batch = search_batch_images.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = self.update_prototypes_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    # form a dict with {class:[images_idxs]}
    class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
    for img_index, img_y in enumerate(search_labels):
        img_y.tolist()
        for idx, i in enumerate(img_y):
            if i:
                class_to_img_index_dict[idx].append(img_index)

    prototype_shape = self.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    # max_dist is chosen arbitrarly
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(self.prototype_class_identity[j]).item()
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]

        # if the smallest distance in the batch is less than the global smallest distance for
        # this prototype
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            # find arguments of the smallest distance in the matrix shape
            arg_min_flat = np.argmin(proto_dist_j)
            arg_min_matrix = np.unravel_index(arg_min_flat, proto_dist_j.shape)
            batch_argmin_proto_dist_j = list(arg_min_matrix)

            # change the index of the smallest distance from the class specific index to the
            # whole search batch index
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][
                batch_argmin_proto_dist_j[0]
            ]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch (the input image e.g. 224x224)
            # that generates the representation
            protoL_rf_info = self.proto_layer_rf_info
            rf_prototype_j = self.compute_rf_prototype(
                search_batch.size(2),
                batch_argmin_proto_dist_j,
                protoL_rf_info,
            )

            # get the whole image
            original_img_j = search_batch_images[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # crop out the receptive field
            rf_img_j = original_img_j[
                rf_prototype_j[1] : rf_prototype_j[2], rf_prototype_j[3] : rf_prototype_j[4], :
            ]

            # save the prototype receptive field information (pixel indices in the input image)
            proto_rf_boxes[j] = {}
            proto_rf_boxes[j]['image_index'] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j]['height_start_index'] = rf_prototype_j[1]
            proto_rf_boxes[j]['height_end_index'] = rf_prototype_j[2]
            proto_rf_boxes[j]['width_start_index'] = rf_prototype_j[3]
            proto_rf_boxes[j]['width_end_index'] = rf_prototype_j[4]
            proto_rf_boxes[j]['class_indentities'] = search_labels[rf_prototype_j[0]].tolist()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            # the activation function of the distances is log
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + self.epsilon))
            # upsample the matrix with distances (e.g., (14x14)->(224x224))
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j,
                dsize=(original_img_size, original_img_size),
                interpolation=cv2.INTER_CUBIC,
            )
            # find a high activation ROI (default treshold = 95 %)
            proto_bound_j = self.find_high_activation_crop(upsampled_act_img_j)
            # crop out the ROI with high activation from the image where the distnce for j
            # protoype turned out to be the smallest
            # the dimensions' order of original_img_j, e.g., (224, 224, 3)
            proto_img_j = original_img_j[
                proto_bound_j[0] : proto_bound_j[1], proto_bound_j[2] : proto_bound_j[3], :
            ]

            # save the ROI (rectangular boundary of highly activated region)
            # the activated region can be larger than the receptive field of the patch with the
            # smallest distance
            proto_bound_boxes[j] = {}
            proto_bound_boxes[j]['image_index'] = proto_rf_boxes[j]['image_index']
            proto_bound_boxes[j]['height_start_index'] = proto_bound_j[0]
            proto_bound_boxes[j]['height_end_index'] = proto_bound_j[1]
            proto_bound_boxes[j]['width_start_index'] = proto_bound_j[2]
            proto_bound_boxes[j]['width_end_index'] = proto_bound_j[3]
            proto_bound_boxes[j]['class_indentities'] = search_labels[rf_prototype_j[0]].tolist()

            # SAVING BLOCK (can be changed later)
            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype activation map (e.g., (14x14))
                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_self_act_filename_prefix + str(j) + '.npy',
                        ),
                        proto_act_img_j,
                    )

                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix + '-original' + str(j) + '.png',
                        ),
                        original_img_j,
                        cmap='gray',
                    )

                    # overlay (upsampled) activation on original image and save the result
                    # normalize the image
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * rescaled_act_img_j),
                        cv2.COLORMAP_JET,
                    )
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix
                            + '-original_with_self_act'
                            + str(j)
                            + '.png',
                        ),
                        overlayed_original_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )

                    # if different from the original (whole) image, save the prototype receptive
                    # field as png
                    if (
                        rf_img_j.shape[0] != original_img_size
                        or rf_img_j.shape[1] != original_img_size
                    ):
                        plt.imsave(
                            os.path.join(
                                dir_for_saving_prototypes,
                                prototype_img_filename_prefix
                                + '-receptive_field'
                                + str(j)
                                + '.png',
                            ),
                            rf_img_j,
                            vmin=0.0,
                            vmax=1.0,
                        )
                        overlayed_rf_img_j = overlayed_original_img_j[
                            rf_prototype_j[1] : rf_prototype_j[2],
                            rf_prototype_j[3] : rf_prototype_j[4],
                        ]
                        plt.imsave(
                            os.path.join(
                                dir_for_saving_prototypes,
                                prototype_img_filename_prefix
                                + '-receptive_field_with_self_act'
                                + str(j)
                                + '.png',
                            ),
                            overlayed_rf_img_j,
                            vmin=0.0,
                            vmax=1.0,
                        )

                    # save the highly activated ROI (highly activated region of the whole image)
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix + str(j) + '.png',
                        ),
                        proto_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )


class SqueezeNet(nn.Module):
    """SqueezeNet encoder.

    The pre-trained model expects input images normalized in the same way, i.e. mini-batches of
    3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The
    images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """

    def __init__(
        self,
        preprocessed: bool = False,
        pretrained: bool = True,
    ):
        """SqueezeNet encoder.

        Args:
            preprocessed (bool, optional): _description_. Defaults to True.
            pretrained (bool, optional): _description_. Defaults to True.
        """

        super().__init__()

        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True, verbose=False
        )
        del self.model.classifier

        self.preprocessed = preprocessed

    def forward(self, x) -> torch.Tensor:
        """Forward implementation.

        Uses only the model.features component of SqueezeNet without model.classifier.

        Args:
            x: raw input in format (batch, channels, spatial, spatial)

        Returns:
            torch.Tensor: convolution layers after passing the SqueezNet backbone
        """
        if self.preprocessed:
            x = self.preprocess(x)

        return self.model.features(x)

    def preprocess(self, x):
        """Image preprocessing function.

        To make image preprocessing model specific and modular.

        Args:
            x: input image.

        Returns:
            preprocessed image.
        """

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return preprocess(x)

    def conv_info(self):
        features = {}
        features['kernel_sizes'] = []
        features['strides'] = []
        features['paddings'] = []

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                if isinstance(module.kernel_size, tuple):
                    features['kernel_sizes'].append(module.kernel_size[0])
                else:
                    features['kernel_sizes'].append(module.kernel_size)

                if isinstance(module.stride, tuple):
                    features['strides'].append(module.stride[0])
                else:
                    features['strides'].append(module.stride)

                if isinstance(module.padding, tuple):
                    features['paddings'].append(module.padding[0])
                else:
                    features['paddings'].append(module.padding)

        return features

    def warm(self) -> None:
        self.requires_grad_(False)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


ENCODERS = {'squezeenet': SqueezeNet()}


class TransientLayers(nn.Sequential):
    def __init__(self, encoder: nn.Module, prototype_shape: List = [9, 512, 1, 1]):
        super().__init__(*_make_layers(encoder, prototype_shape))

    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


class PrototypeLayer(nn.Parameter):
    def __init__(
        self, num_classes: int, num_prototypes: int, prototype_shape: Sequence[int], prototype_layer_stride: int = 1
    ):
        super().__init__()
        self._num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.num_prototypes_per_class = self.num_prototypes // self._num_classes
        self.shape = prototype_shape
        self.layer_stride = prototype_layer_stride

    def update(self, model: pl.LightningModule, dataloader: DataLoader, updater: ProtoUpdater, saver: Optional[ImageSaver] = None) -> None:
        self.updater.update_prototypes(model, dataloader, saver)

    def compute_proto_layer_rf_info(self, img_size: int, conv_info: Dict[str, int]) -> List[Union[int, float]]:
        _check_dimensions(conv_info)
        # receptive field parameters for the first layer (image itself)
        rf_info = [img_size, 1, 1, 0.5]
        rf_info = _extract_network_rf_info(conv_info, rf_info)
        proto_layer_rf_info = _compute_layer_rf_info(
            layer_filter_size=self.shape[2],
            layer_stride=self.layer_stride,
            layer_padding='VALID',
            previous_layer_rf_info=rf_info,
        )
        return proto_layer_rf_info

    def make_prototype_class_identity(self):
        # onehot indication matrix for prototypes (num_prototypes, num_classes)        
        prototype_class_identity = torch.zeros(
            self.num_prototypes,
            self._num_classes,
            dtype=torch.float,
        ).cuda()
        # fills with 1 only those prototypes, which correspond to the correct class. The rest is
        # filled with 0
        for i in range(self.num_prototypes):
            prototype_class_identity[i, i // self.num_prototypes_per_class] = 1
        return prototype_class_identity

    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(False)


class LastLayer(nn.Linear):
    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(True)


def _check_dimensions(conv_info: Dict[str, int]) -> None:
    if len(conv_info['kernel_sizes']) != len(conv_info['strides']):
        raise Exception("The number of kernels has to be equla to the number of strides")
    if len(conv_info['kernel_sizes']) != len(conv_info['paddings']):
        raise Exception("The number of kernels has to be equla to the number of paddings")


def _extract_network_rf_info(conv_info: Dict[str, int], rf_info: Sequence[Union[int, float]]) -> List[Union[int, float]]:
    for i in range(len(conv_info['kernel_sizes'])):
        rf_info = _compute_layer_rf_info(
            layer_filter_size=conv_info['kernel_sizes'][i],
            layer_stride=conv_info['strides'][i],
            layer_padding=conv_info['paddings'][i],
            previous_layer_rf_info=rf_info,
        )
    return rf_info


def _compute_layer_rf_info(layer_filter_size: int, layer_stride: int, layer_padding: int, previous_layer_rf_info: List[Union[int, float]]) -> List[Union[int, float]]:
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
