from typing import List, Optional, Sequence, Dict, Union

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pytorch_lightning as pl

from src.models.prototype_model.utils import _make_layers
from utils import ImageSaver, copy_tensor_to_nparray


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
        self, num_prototypes: int, prototype_shape: Sequence[int], prototype_layer_stride: int = 1
    ):
        super().__init__()
        self.stride = prototype_layer_stride
        self.prototype_updater = _PrototypeUpdater(num_prototypes, prototype_shape)

    def update(self, dataloader: DataLoader, saver: Optional[ImageSaver] = None):
        self.prototype_updater.update_prototypes(dataloader, saver)

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


class _PrototypeUpdater:
    """Class for updating prototypes.

    The class performs all the necessary internal work to update protoypes.

    Args:
        model: the pl.LightningModule subclass from which this class is called.
        dataloader: DataLoader with the data.
        proto_rf_boxes: initial dict for storing receptive field boxes of the prototypes.
        proto_bound_boxes: initial dict for storing bound boxes based on the activations of the
            prototypes.
    """

    def __init__(self, num_prototypes: int, prototype_shape: Sequence[int]) -> None:
        self._global_min_proto_dist = _PrototypeUpdater._get_global_min_proto_dist(num_prototypes)
        self._global_min_fmap_patches = _PrototypeUpdater._get_global_min_fmap_patches(
            num_prototypes, prototype_shape
        )
        self._proto_rf_boxes = _PrototypeUpdater._get_proto_rf_boxes()
        self._proto_bound_boxes = _PrototypeUpdater._get_proto_bound_boxes()

    def update_prototypes(self, model: pl.LightningModule, dataloader: DataLoader, saver: Optional[ImageSaver]) -> None:
        with tqdm(total=len(dataloader), desc='Updating prototypes', position=3, leave=False) as t:
            for iter, batch in enumerate(dataloader):
                batch_index = _get_batch_index(iter, dataloader.batch_size)
                self._update_prototypes_on_batch(model, batch, batch_index, saver)

    def _update_prototypes_on_batch(
        self, model: pl.LightningModule, batch: torch.Tensor, batch_index: int, saver: Optional[ImageSaver]
    ):
        images, masks, labels = _split_batch(batch)
        proto_layer_input, proto_distances = self._get_input_output_proto_layer(model, images)

    def _get_input_output_of_proto_layer(self, model: pl.LightningModule, images: torch.Tensor):
        if model.training: model.eval()
        with torch.no_grad():
            search_batch = images.cuda()
            # this computation currently is not parallelized
            proto_layer_input_torch, proto_distances_torch = model.update_prototypes_forward(search_batch)
        proto_layer_input = copy_tensor_to_nparray(proto_layer_input_torch)
        proto_distances = copy_tensor_to_nparray(proto_distances_torch)
        del proto_layer_input_torch, proto_distances_torch
        return proto_layer_input, proto_distances   

    @staticmethod
    def _get_global_min_proto_dist(num_prototypes: int) -> np.array:
        # returns tensor for global per epoch min distances initialized by infs
        return np.full(num_prototypes, np.inf)

    @staticmethod
    def _get_global_min_fmap_patches(
        num_prototypes: int, prototype_shape: Sequence[int]
    ) -> np.array:
        # returns tensor for global per epoch feature maps initialized by zeros
        return np.zeros(
            (num_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]),
        )

    @staticmethod
    def _get_proto_rf_boxes() -> Dict[int, Union[int, Sequence[int]]]:
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
    def _get_proto_bound_boxes() -> Dict[int, Union[int, Sequence[int]]]:
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