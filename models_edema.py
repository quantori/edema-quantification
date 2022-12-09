"""Models for the edema classification project.

The description to be filled...
"""
import os
import math

from turtle import shape
from typing import Tuple
from matplotlib import image
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms, datasets
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SqueezeNet(nn.Module):
    """SqueezeNet encoder.

    The pre-trained model expects input images normalized in the same way, i.e. mini-batches of
    3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The
    images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """

    def __init__(
        self,
        preprocessed: bool = True,
        pretrained: bool = True,
    ):
        """SqueezeNet encoder.

        Args:
            preprocessed (bool, optional): _description_. Defaults to True.
            pretrained (bool, optional): _description_. Defaults to True.
        """

        super().__init__()

        self.model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)
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


class EdemaNet(pl.LightningModule):
    """PyTorch Lightning model class.

    A complete model is implemented (includes the encoder, transient, prototype and fully connected
    layers). The transient layers are required to concatenate the main encoder layers with the
    prototype layer. The encoder is the variable part of EdemaNet, which is passed as an argument.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        prototype_shape: Tuple,
        transient_layers_type: str = "bottleneck",
        top_k: int = 1,
        fine_loader: torch.utils.data.DataLoader = None,
        num_warm_epochs: int = 10,
        push_start: int = 10,
        push_epoch: list = [],
    ):
        """PyTorch Lightning model class.

        Args:
            encoder (nn.Module): encoder layers implemented as a distinct class.
            num_classes (int): the number of feature classes.
            prototype_shape (Tuple): the shape of the prototypes (batch, channels, H, W).
            transient_layers_type (str, optional): the architecture of the transient layers. If ==
                                                   'bottleneck', the number of channels is adjusted
                                                   automatically.
            top_k (int): the number of the closest distances between patches and prototypes to be
                         considered for the similarity score calculation
        """

        super().__init__()

        self.transient_layers_type = transient_layers_type
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape
        self.top_k = top_k  # for a 14x14: top_k=3 is 1.5%, top_k=9 is 4.5%
        self.epsilon = 1e-4  # needed for the similarity calculation
        self.fine_loader = fine_loader
        self.num_classes = num_classes
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_warm_epochs = num_warm_epochs
        self.push_start = push_start
        self.push_epoch = push_epoch
        # cross entropy cost function
        self.cross_entropy_cost = nn.BCEWithLogitsLoss()
        # receptive-field information that is needed to cut out the chosen upsampled fmap patch
        # TODO: kernels, strides, and paddings
        # self.proto_layer_rf_info = self.compute_proto_layer_rf_info(
        #     img_size=img_size,
        #     layer_filter_sizes=layer_filter_sizes,
        #     layer_strides=layer_strides,
        #     layer_paddings=layer_paddings,
        #     prototype_kernel_size=prototype_shape[2],
        # )

        # onehot indication matrix for prototypes (num_prototypes, num_classes)
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes, self.num_classes, dtype=torch.float
        )
        # fills with 1 only those prototypes, which correspond to the correct class. The rest is
        # filled with 0
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        # encoder
        self.encoder = encoder

        # transient layers
        self.transient_layers = self._make_transient_layers(self.encoder)

        # prototypes layer (do not make this just a tensor, since it will not be moved
        # automatically to gpu)
        self.prototype_layer = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # last fully connected layer for the classification of edema features. The bias is not used
        # in the original paper
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

    def forward(self, x):
        # x is a batch of images having (batch, 3, H, W) dimensions
        if x.shape[1] != 3:
            raise Exception("The channel dimension of the input-image batch has to be 3")

        # save the spatial dimensions of the initial image to use them later for upsampling (PAM
        # creation)
        upsample_hight, upsample_width = x.shape[2], x.shape[3]

        x = self.encoder(x)
        x = self.transient_layers(x)
        distances = self.prototype_distances(x)

        _distances = distances.view(distances.shape[0], distances.shape[1], -1)
        # in topk(), if dim is not given, the last dimension of the input is chosen
        closest_k_distances, _ = torch.topk(_distances, self.top_k, largest=False)
        min_distances = F.avg_pool1d(
            closest_k_distances, kernel_size=closest_k_distances.shape[2]
        ).view(-1, self.num_prototypes)

        prototype_activations = torch.log((distances + 1) / (distances + self.epsilon))
        _activations = prototype_activations.view(
            prototype_activations.shape[0], prototype_activations.shape[1], -1
        )
        top_k_activations, _ = torch.topk(_activations, self.top_k)
        prototype_activations = F.avg_pool1d(
            top_k_activations, kernel_size=top_k_activations.shape[2]
        ).view(-1, self.num_prototypes)

        logits = self.last_layer(prototype_activations)

        activation = torch.log((distances + 1) / (distances + self.epsilon))
        upsampled_activation = torch.nn.Upsample(
            size=(upsample_hight, upsample_width), mode="bilinear", align_corners=False
        )(activation)

        return logits, min_distances, upsampled_activation

    def update_prootypes_forward(self, x):
        """This method is needed for the prototype updating operation"""
        conv_output = self.encoder(x)
        conv_output = self.transient_layers(conv_output)
        distances = self.prototype_distances(conv_output)
        return conv_output, distances

    def training_step(self, batch, batch_idx):
        # based on universal train_val_test(). Logs costs after each train step
        cost = self.train_val_test(batch)
        return cost

    def on_train_epoch_start(self):
        if self.current_epoch < self.num_warm_epochs:
            self.warm_only()
        else:
            self.joint()

    def training_epoch_end(self, outputs):
        pass
        # # here, we have to put push_prototypes function
        # # logs costs after a training epoch
        # if self.current_epoch >= self.push_start and self.current_epoch in self.push_epochs:

        #     # TODO implement update_prototype() func
        #     self.update_prototypes()

        #     # has to be test (to check out the performance after substituting the prototypes)
        #     accu = self.train_val_test()

        #     self.last_layer()
        #     for i in range(10):
        #         # has to be train
        #         self.train_val_test()

        #         # has to be test
        #         self.train_val_test

        #         # save model performance

        #         # calculate performance and update the global performance criterium, if it is worse

        #         # optionally (plot something)

    def test_step(self, batch, batch_idx):
        # this is for testing after training and validation are done
        pass

    def test_epoch_end(self, outputs):
        pass

    def train_val_test(self, batch):
        images, labels = batch
        # labels - (batch, 7), dtype: float32
        # images - (batch, 10, H, W)
        # images have to have the shape (batch, 10, H, W). 7 extra channel implies fine annotations,
        # in case of 7 classes
        if images.shape[1] != 10:
            raise Exception("The channel dimension of the input-image batch has to be 10")

        fine_annotations = images[:, 3:10, :, :]  # 7 classes of fine annotations
        images = images[:, 0:3, :, :]  # (no view, create slice)

        # nn.Module has implemented __call__() function, so no need to call .forward()
        output, min_distances, upsampled_activation = self(images)

        # classification cost (labels in cross entropy have to have float32 dtype)
        cross_entropy = self.cross_entropy_cost(output, labels)

        # cluster cost
        max_dist = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        prototypes_of_correct_class = torch.matmul(
            labels, self.prototype_class_identity.permute(1, 0)
        )
        cluster_cost = self.cluster_cost(max_dist, min_distances, prototypes_of_correct_class)

        # separation cost
        separation_cost = self.separation_cost(max_dist, min_distances, prototypes_of_correct_class)

        # fine cost
        fine_cost = self.fine_cost(
            fine_annotations, upsampled_activation, self.num_prototypes_per_class
        )

        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 0.001 * fine_cost

        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        return loss

    def warm_only(self):
        self.encoder.requires_grad_(False)
        self.transient_layers.requires_grad_(True)
        self.prototype_layer.requires_grad_(True)
        self.last_layer.requires_grad_(True)

    def last_only(self):
        self.encoder.requires_grad_(False)
        self.transient_layers.requires_grad_(False)
        self.prototype_layer.requires_grad_(False)
        self.last_layer.requires_grad_(True)

    def joint(self):
        self.encoder.requires_grad_(True)
        self.transient_layers.requires_grad_(True)
        self.prototype_layer.requires_grad_(True)
        self.last_layer.requires_grad_(True)

    def configure_optimizers(self):
        # TODO configure the optimizer properly
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def prototype_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Returns prototype distances.

        Slices through the image in the latent space and calculates a p-norm distance between the
        image patches and prototypes.

        Args:
            x (torch.Tensor): image after convolution layers

        Returns:
            torch.Tensor: prototype distances of size (batch, num_prototypes, conv output shape)
        """

        # x is the conv output, shape=[Batch * channel * conv output shape]
        batch = x.shape[0]

        # if the input is (batch, 512, 14, 14) and the kernel_size=(1, 1), the output will be
        # (batch, 512=512*1*1, 196=14*14)
        expanded_x = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(x)

        # expanded shape = [1, batch, number of such blocks, channel*proto_shape[2]*proto_shape[3]]
        expanded_x = expanded_x.unsqueeze(0).permute(0, 1, 3, 2)

        # change the input tensor into contiguous in memory tensor (make a copy). The output of the
        # view() (if input=(1, batch, 196, 512)) -> a tensor of (1, batch*196, 512=512*1*1)
        # dimension (if the prototype dimensions are (num_prototypes, 512, 1, 1)). -1 means that
        # this dimension is calculated based on other dimensions
        expanded_x = expanded_x.contiguous().view(
            1, -1, self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        )

        # if the input tensor is (num_prototypes, 512, 1, 1) and the kernel_size=(1, 1), the output
        # is (1, num_prototypes, 512=512*1*1, 1=1*1).
        # Expanded proto shape = [1, proto num, channel*proto_shape[2]*proto_shape[3], 1]
        expanded_proto = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(
            self.prototype_layer
        ).unsqueeze(0)

        # if the input is (1, num_prototypes, 512, 1), the output is (1, num_prototypes, 512=512*1)
        expanded_proto = expanded_proto.contiguous().view(1, expanded_proto.shape[1], -1)

        # the output is (1, Batch * number of blocks in x, num_prototypes). If expanded_x is
        # (1, batch*196, 512) and expanded_proto is (1, num_prototypes, 512), the output shape is
        # (1, batch*196, num_prototypes). The default p value for the p-norm distance = 2 (euclidean
        # distance)
        expanded_distances = torch.cdist(expanded_x, expanded_proto)

        # (1, batch*196, num_prototypes) -> (batch, num_prototypes, 1*196)
        expanded_distances = torch.reshape(
            expanded_distances, shape=(batch, -1, self.prototype_shape[0])
        ).permute(0, 2, 1)

        # print(expanded_distances.shape)
        # distances = nn.Fold(
        #     output_size=(
        #         x.shape[2] - self.prototype_shape[2] + 1,
        #         x.shape[3] - self.prototype_shape[3] + 1,
        #     ),
        #     kernel_size=(self.prototype_shape[2], self.prototype_shape[3]),
        # )(expanded_distances)

        # distance shape = (batch, num_prototypes, conv output shape)
        distances = torch.reshape(
            expanded_distances,
            shape=(
                batch,
                self.prototype_shape[0],
                x.shape[2] - self.prototype_shape[2] + 1,
                x.shape[3] - self.prototype_shape[3] + 1,
            ),
        )

        return distances

    def cluster_cost(
        self,
        max_dist: int,
        min_distances: torch.Tensor,
        prototypes_of_correct_class: torch.Tensor,
    ) -> torch.Tensor:
        """Returns cluster cost.

        Args:
            max_dist (int): max distance is needed for the inverting trick (have a look below)
            min_distances (torch.Tensor): min distances returned by the model.forward()
            prototypes_of_correct_class (torch.Tensor): (batch, num_prototypes)
            labels (torch.Tensor): batch of labels taken from a dataloader

        Returns:
            torch.Tensor: cluster cost of size (1,) makes prototypes of the same class closer to
                          each other
        """
        # (max_dist - min_distances) and, consequently, inverted_distances is done to prevent the
        # constant retrieving of 0 for min values obtained by *prototypes_of_correct_class
        inverted_distances, _ = torch.max(
            (max_dist - min_distances) * prototypes_of_correct_class, dim=1
        )
        cluster_cost = torch.mean(max_dist - inverted_distances)

        return cluster_cost

    def separation_cost(
        self,
        max_dist: int,
        min_distances: torch.Tensor,
        prototypes_of_correct_class: torch.Tensor,
    ) -> torch.Tensor:
        """Returns separation cost.

        Args:
            max_dist (int): max distance is needed for the inverting trick (have a look below)
            min_distances (torch.Tensor): min distances returned by the model.forward()
            prototypes_of_correct_class (torch.Tensor): (batch, num_prototypes)
            labels (torch.Tensor): batch of labels taken from a dataloader

        Returns:
            torch.Tensor: separation cost of size (1,) makes prototypes of different classes farer
                          from each other
        """
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        # (max_dist - min_distances) and, consequently, inverted_distances is done to prevent the
        # constant retrieving of 0 for min values obtained by *prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = torch.max(
            (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
        )
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        return separation_cost

    def fine_cost(
        self,
        fine_annotations: torch.Tensor,
        upsampled_activations: torch.Tensor,
        num_prototypes_per_class: int,
    ) -> torch.Tensor:
        """Returns fine cost.

        Args:
            fine_annotations (torch.Tensor): (batch, num_classes, H, W)
            upsampled_activations (torch.Tensor): (batch, num_prototypes, H, W)
            num_prototypes_per_class (int): number of prototypes per class

        Returns:
            torch.Tensor: fine cost of size (1,) forces activating the prototypes in the
                          mask-allowed regions (fine annotations)
        """
        fine_annotations = torch.repeat_interleave(fine_annotations, num_prototypes_per_class, 1)
        fine_cost = torch.norm(upsampled_activations * fine_annotations)

        return fine_cost

    def _make_transient_layers(self, encoder: torch.nn.Module) -> torch.nn.Sequential:
        """Returns transient layers.

        Args:
            encoder (torch.nn.Module): encoder architecture.

        Returns:
            torch.nn.Sequential: transient layers as the PyTorch Sequential class.
        """
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
        if self.transient_layers_type == "bottleneck":
            transient_layers = []
            current_in_channels = first_transient_layer_in_channels

            while (current_in_channels > self.prototype_shape[1]) or (len(transient_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
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

                if current_out_channels > self.prototype_shape[1]:
                    transient_layers.append(nn.ReLU())

                else:
                    assert current_out_channels == self.prototype_shape[1]
                    transient_layers.append(nn.Sigmoid())

                current_in_channels = current_in_channels // 2

            transient_layers = nn.Sequential(*transient_layers)

            return transient_layers

        # determined transient layers
        else:
            transient_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_transient_layer_in_channels,
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.prototype_shape[1],
                    out_channels=self.prototype_shape[1],
                    kernel_size=1,
                ),
                nn.Sigmoid(),
            )

            return transient_layers

    def compute_layer_rf_info(self, layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
        # based on https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        n_in = previous_layer_rf_info[0] # receptive-field input size
        j_in = previous_layer_rf_info[1] # receptive field jump of input layer
        r_in = previous_layer_rf_info[2] # receptive field size of input layer
        start_in = previous_layer_rf_info[3] # center of receptive field of input layer

        if layer_padding == 'SAME':
            n_out = math.ceil(float(n_in) / float(layer_stride))
            if (n_in % layer_stride == 0):
                pad = max(layer_filter_size - layer_stride, 0)
            else:
                pad = max(layer_filter_size - (n_in % layer_stride), 0)
            assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
            assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
        elif layer_padding == 'VALID':
            n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
            pad = 0
            assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
            assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
        else:
            # layer_padding is an int that is the amount of padding on one side
            pad = layer_padding * 2
            n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

        pL = math.floor(pad/2)

        j_out = j_in * layer_stride
        r_out = r_in + (layer_filter_size - 1)*j_in
        start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
        
        return [n_out, j_out, r_out, start_out]


    def compute_proto_layer_rf_info(
        self, img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size
    ):
        # TODO: implement kernel_sizes = [...], strides = [...], paddings = [...] in the encoder
        # class
        if len(layer_filter_sizes) != len(layer_strides):
            raise Exception("The number of kernels has to be equla to the number of strides")
        if len(layer_filter_sizes) != len(layer_paddings):
            raise Exception("The number of kernels has to be equla to the number of paddings")

        # receptive field parameters for the first layer (image itself)
        rf_info = [img_size, 1, 1, 0.5]

        for i in range(len(layer_filter_sizes)):
            filter_size = layer_filter_sizes[i]
            stride_size = layer_strides[i]
            padding_size = layer_paddings[i]

            rf_info = self.compute_layer_rf_info(
                layer_filter_size=filter_size,
                layer_stride=stride_size,
                layer_padding=padding_size,
                previous_layer_rf_info=rf_info,
            )

            proto_layer_rf_info = self.compute_layer_rf_info(
                layer_filter_size=prototype_kernel_size,
                layer_stride=1,
                layer_padding='VALID',
                previous_layer_rf_info=rf_info,
            )

            return proto_layer_rf_info

    def update_prototypes(self, root_dir_for_saving_prototypes):
        self.eval()
        prototype_shape = self.prototype_shape
        n_prototypes = self.num_prototypes
        # train dataloader
        dataloader = self.trainer.train_dataloader.loaders

        # make an array for the global closest distance seen so far (initialized with floating point
        # representation of positive infinity)
        global_min_proto_dist = np.full(n_prototypes, np.inf)

        # saves the patch representation that gives the current smallest distance
        global_min_fmap_patches = np.zeros(
            [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
        )

        # proto_rf_boxes (receptive field) and proto_bound_boxes column:
        # 0: image index in the entire dataset
        # 1: height start index
        # 2: height end index
        # 3: width start index
        # 4: width end index
        # 5: class identity
        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)

        # making a directory for saving prototypes
        if root_dir_for_saving_prototypes != None:
            if self.current_epoch != None:
                proto_epoch_dir = os.path.join(
                    root_dir_for_saving_prototypes, 'epoch-' + str(self.current_epoch)
                )
                if not os.path.exists(proto_epoch_dir):
                    os.makedirs(proto_epoch_dir)
            else:
                proto_epoch_dir = root_dir_for_saving_prototypes
        else:
            proto_epoch_dir = None

        search_batch_size = dataloader.batch_size
        num_classes = self.num_classes

        for push_iter, (search_batch_images, search_labels) in enumerate(dataloader):
            if search_batch_images.shape[1] > 3:
                # only imagees (the extra channels in this dimension belong to fine annot masks)
                search_batch_images = search_batch_images[:, 0:3, :, :]

            start_index_of_search_batch = push_iter * search_batch_size

            self.update_prototypes_on_batch(search_batch_images, search_labels)

    def update_prototypes_on_batch(
        self,
        search_batch_images,
        search_labels,
        global_min_proto_dist,
        global_min_fmap_patches,
        prototype_layer_stride=1,
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

        # TODO: finsih the cycle
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

                # get the receptive field boundary of the image patch
                # that generates the representation
                protoL_rf_info = self.proto_layer_rf_info
                rf_prototype_j = compute_rf_prototype(
                    search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info
                )


if __name__ == "__main__":

    sq_net = SqueezeNet()
    # summary(sq_net.model, (3, 224, 224))
    edema_net = EdemaNet(sq_net, 7, prototype_shape=(35, 512, 1, 1))

    test_dataset = TensorDataset(
        torch.rand(128, 10, 224, 224), torch.randint(0, 2, (128, 7), dtype=torch.float32)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    batch = next(iter(test_dataloader))
    images, labels = batch

    n_in = 11
    layer_filter_size = 3
    layer_stride = 2
    n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
    print(n_out)
    pad = 0

    assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
    # assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check

    # class_to_img_index_dict = {key: [] for key in range(7)}
    # for img_index, img_y in enumerate(labels):
    #     img_y.tolist()
    #     for idx, i in enumerate(img_y):
    #         if i:
    #             class_to_img_index_dict[idx].append(img_index)
    # print(class_to_img_index_dict)
    # batch[0].cuda
    # print(batch[0].is_cuda)
    # print(torch.__version__)

    # print(edema_net.training)
    # edema_net.eval()
    # print(edema_net.training)

    # print(list(edema_net.named_parameters())[0][1].requires_grad)
    # for name, param in edema_net.named_parameters():
    # print(name, 'requires_grad: ', param[1].requires_grad)

    # print(edema_net.trainer.train_dataloader)

    # trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, gpus=1)
    # trainer.fit(edema_net, test_dataloader)

    # for name, param in edema_net.named_parameters():
    # print(name, 'requires_grad: ', param[1].requires_grad)

    # use it for the prototype pushing function
    # print(type(edema_net.trainer.train_dataloader.loaders))
    # print(edema_net.trainer.train_dataloader.loaders.batch_size)
