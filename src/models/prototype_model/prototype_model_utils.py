from typing import Any, Dict, Optional, Union, List, Sequence, Tuple, Callable, Iterable
import sys
import math
from dataclasses import dataclass

from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import numpy as np

from models_edema import EdemaNet


class PNetProgressBar(TQDMProgressBar):
    def __init__(self, process_position: int = 1):
        super().__init__(process_position=process_position)
        self._status_bar: Optional[tqdm] = None

    @property
    def status_bar(self) -> tqdm:
        if self._status_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._status_bar` reference has not been set yet."
            )
        return self._status_bar

    @status_bar.setter
    def status_bar(self, bar: tqdm) -> None:
        self._status_bar = bar

    def init_status_tqdm(self) -> tqdm:
        bar = tqdm(total=0, position=1, bar_format='{desc}', file=sys.stdout)
        return bar

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self.status_bar = self.init_status_tqdm()


def get_encoder(encoders: dict, name: str = 'squezeenet') -> nn.Module:
    try:
        return encoders[name]
    except:
        print(f'{name} encoder is not implemented')


def _compute_layer_rf_info(
    layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info
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


def compute_proto_layer_rf_info(
    img_size: int, conv_info: Dict[str, int], prototype_kernel_size: int
) -> List[Union[int, float]]:
    check_dimensions(conv_info=conv_info)

    # receptive field parameters for the first layer (image itself)
    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(conv_info['kernel_sizes'])):
        filter_size = conv_info['kernel_sizes'][i]
        stride_size = conv_info['strides'][i]
        padding_size = conv_info['paddings'][i]

        rf_info = _compute_layer_rf_info(
            layer_filter_size=filter_size,
            layer_stride=stride_size,
            layer_padding=padding_size,
            previous_layer_rf_info=rf_info,
        )

    proto_layer_rf_info = _compute_layer_rf_info(
        layer_filter_size=prototype_kernel_size,
        layer_stride=1,
        layer_padding='VALID',
        previous_layer_rf_info=rf_info,
    )

    return proto_layer_rf_info


def check_dimensions(conv_info: Dict[str, int]) -> None:
    if len(conv_info['kernel_sizes']) != len(conv_info['strides']):
        raise Exception("The number of kernels has to be equla to the number of strides")
    if len(conv_info['kernel_sizes']) != len(conv_info['paddings']):
        raise Exception("The number of kernels has to be equla to the number of paddings")


def _make_layers(
    encoder: nn.Module, prototype_shape: Sequence[Union[int, float]]
) -> List[nn.Module]:
    if encoder.__class__.__name__ == "SqueezeNet":
        first_transient_layer_in_channels = (
            2 * [i for i in encoder.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        )
    else:
        first_transient_layer_in_channels = [
            i for i in encoder.modules() if isinstance(i, nn.Conv2d)
        ][-1].out_channels

    # automatic adjustment of the transient-layer channels for matching with the prototype
    # channels. The activation functions of the intermediate and last transient layers are ReLU
    # and sigmoid, respectively
    # if self.transient_layers_type == "bottleneck":
    transient_layers = []
    current_in_channels = first_transient_layer_in_channels

    while (current_in_channels > prototype_shape[1]) or (len(transient_layers) == 0):
        current_out_channels = max(prototype_shape[1], (current_in_channels // 2))
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            )
        )
        transient_layers.append(nn.ReLU())
        transient_layers.append(
            nn.Conv2d(
                in_channels=current_out_channels,
                out_channels=current_out_channels,
                kernel_size=1,
            )
        )

        if current_out_channels > prototype_shape[1]:
            transient_layers.append(nn.ReLU())

        else:
            assert current_out_channels == prototype_shape[1]
            transient_layers.append(nn.Sigmoid())

        current_in_channels = current_in_channels // 2

    return transient_layers


def warm(**kwargs) -> None:
    for arg in kwargs.values():
        arg.warm()


def joint(**kwargs) -> None:
    for arg in kwargs.values():
        arg.joint()


def last(**kwargs) -> None:
    for arg in kwargs.values():
        arg.last()


def print_status_bar(trainer: pl.Trainer, blocks: Dict, status: str = '') -> None:
    trainer.progress_bar_callback.status_bar.set_description_str(
        '{status}, REQUIRES GRAD: Encoder ({encoder}), Transient layers ({trans}),'
        ' Protorype layer ({prot}), Last layer ({last})'.format(
            status=status,
            encoder=get_grad_status(blocks['encoder']),
            trans=get_grad_status(blocks['transient_layers']),
            # protoype layer is inhereted from Tensor and, therefore, has the requires_grad attr
            prot=blocks['prototype_layer'].requires_grad,
            last=get_grad_status(blocks['last_layer']),
        )
    )


def get_grad_status(block: nn.Module) -> bool:
    first_param = next(block.parameters())
    if all(param.requires_grad == first_param.requires_grad for param in block.parameters()):
        return first_param.requires_grad
    else:
        raise Exception(
            f'Not all the parmaters in {block.__class__.__name__} have the same grad status'
        )


class _Batch:
    """Internal class for storing batch used in update_prototypes func.

    Attributes:
        images_and_masks: images + fine annotation masks obtained from a batch.
        images: images separated from the masks.
        labels: labels obtained from the batch.
        index: the index of the current batch.
    """
    def __init__(self, batch: torch.Tensor, iter: int, batch_size: int) -> None:
        self.images_and_masks = batch[0]
        self.images = self.images_and_masks[:, 0:3, :, :]
        self.labels = batch[1]
        self.index = iter * batch_size


class PrototypeUpdater:
    """Class for updating prototypes.

    The class performs all the necessary internal work to update protoypes.

    Args:
        model: the pl.LightningModule subclass from which this class is called. 
        dataloader: DataLoader with the data.
        proto_rf_boxes: initial dict for storing receptive field boxes of the prototypes.
        proto_bound_boxes: initial dict for storing bound boxes based on the activations of the
            prototypes.
    """
    def __init__(self, model: EdemaNet, dataloader: DataLoader, settings_save: Optional[DictConfig] = None) -> None:
        self._model = model
        self._dataloader = dataloader
        if settings_save is not None:
            self._settings_save = settings_save
        self._global_min_proto_dist = self._get_global_min_proto_dist(model.num_prototypes)
        self._global_min_fmap_patches = self._get_global_min_fmap_patches(model.num_prototypes, model.prototype_shape)
        self._proto_rf_boxes = self._get_proto_rf_boxes()
        self._proto_bound_boxes = self._get_proto_bound_boxes()

    def update_prototypes(self) -> None:
        with tqdm(total=len(self.dataloader), desc='Updating prototypes', position=3, leave=False) as t:
            for push_iter, batch in enumerate(self.dataloader):
                self._update_prototypes_on_batch()
                
    def _update_prototypes_on_batch(self):
        pass

        
    def _get_global_min_proto_dist(self, num_prototypes: int) -> np.array:
        # returns tensor for global per epoch min distances initialized by infs
        return np.full(num_prototypes, np.inf)

    def _get_global_min_fmap_patches(self, num_prototypes: int, prototype_shape: Iterable[int]) -> np.array:
        # returns tensor for global per epoch feature maps initialized by zeros
        return np.zeros((num_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]),)

    def _get_proto_rf_boxes(self) -> Dict[int, Union[int, Iterable[int]]]:
        # initial dict for storing receptive field boxes of the prototypes. It is supposed to have
        # the following structure:
        #     0: image index in the entire dataset
        #     1: height start index
        #     2: height end index
        #     3: width start index
        #     4: width end index
        #     5: class identities
        return {}

    def _get_proto_bound_boxes(self) -> Dict[int, Union[int, Iterable[int]]]:
        # initial dict for storing bound boxes based on the activations of the prototypes. It is
        # suposed to have the same structure as proto_rf_boxes
        return {}

def update_prototypes(
    model: EdemaNet,
    dataloader: DataLoader,  # pytorch dataloader (must be unnormalized in [0,1])
    settings_save: DictConfig = None
    # prototype_layer_stride: int = 1,
    # root_dir_for_saving_prototypes: str = './savings/',  # if not None, prototypes will be saved here
    # prototype_img_filename_prefix: str = 'prototype-img',
    # prototype_self_act_filename_prefix: str = 'prototype-self-act',
    # proto_bound_boxes_filename_prefix: str = 'bb',
) -> None:

    # making a directory for saving prototypes
    # if root_dir_for_saving_prototypes != None:
    #     if self.current_epoch != None:
    #         proto_epoch_dir = os.path.join(
    #             root_dir_for_saving_prototypes,
    #             'epoch-' + str(self.current_epoch),
    #         )
    #         if not os.path.exists(proto_epoch_dir):
    #             os.makedirs(proto_epoch_dir)
    #     else:
    #         proto_epoch_dir = root_dir_for_saving_prototypes
    # else:
    #     proto_epoch_dir = None

    # search_batch_size = dataloader.batch_size

    with tqdm(total=len(dataloader), desc='Updating prototypes', position=3, leave=False) as t:
        for push_iter, batch in enumerate(dataloader):
            update_prototypes_on_batch(
                _Batch(batch, push_iter, dataloader.batch_size),
                _GlobalActivations(model.num_prototypes, model.prototype_shape), 
                prototype_layer_stride=model.prototype_layer.stride,
                dir_for_saving_prototypes=proto_epoch_dir,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            )
            t.update()

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        proto_rf_boxes_json = json.dumps(proto_rf_boxes)
        f = open(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix
                + '-receptive_field'
                + str(self.current_epoch)
                + '.json',
            ),
            'w',
        )
        f.write(proto_rf_boxes_json)
        f.close()

        proto_bound_boxes_json = json.dumps(proto_bound_boxes)
        f = open(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix + str(self.current_epoch) + '.json',
            ),
            'w',
        )
        f.write(proto_bound_boxes_json)
        f.close()

    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    self.prototype_layer.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # # prototype_network_parallel.cuda()
    # end = time.time()
    # log('\tpush time: \t{0}'.format(end - start))


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
