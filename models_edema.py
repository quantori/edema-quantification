"""Models for the edema classification project.

The description to be filled...
"""
import os
import math
import json
from typing import Optional

from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torchmetrics.functional.classification import multilabel_f1_score


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
        num_warm_epochs: int = 2,
        push_start: int = 2,
        push_epochs: list = [2, 4, 6],
        img_size=224,
    ):
        """PyTorch Lightning model class.

        Args:
            encoder (nn.Module): encoder layers implemented as a distinct class.
            num_classes (int): the number of feature classes.
            prototype_shape (Tuple): the shape of the prototypes (num_prototypes, channels, H, W).
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
        self.push_epochs = push_epochs
        # cross entropy cost function
        self.cross_entropy_cost = nn.BCEWithLogitsLoss()

        # receptive-field information that is needed to cut out the chosen upsampled fmap patch
        kernel_sizes = encoder.conv_info()['kernel_sizes']
        layer_strides = encoder.conv_info()['strides']
        layer_paddings = encoder.conv_info()['paddings']

        self.proto_layer_rf_info = self.compute_proto_layer_rf_info(
            img_size=img_size,
            layer_filter_sizes=kernel_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=prototype_shape[2],
        )

        # onehot indication matrix for prototypes (num_prototypes, num_classes)
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes, self.num_classes, dtype=torch.float
        ).cuda()
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

    def update_prototypes_forward(self, x):
        """This method is needed for the prototype updating operation"""
        conv_output = self.encoder(x)
        conv_output = self.transient_layers(conv_output)
        distances = self.prototype_distances(conv_output)
        return conv_output, distances

    def training_step(self, batch, batch_idx):
        # based on universal train_val_test(). Logs costs after each train step
        cost, f1_score = self.train_val_test(batch)
        self.log('f1_score_train', f1_score, prog_bar=True)
        self.log('train_cost', cost, prog_bar=True)
        return cost

    def on_train_epoch_start(self):
        if self.current_epoch < self.num_warm_epochs:
            self.warm_only()
        else:
            self.joint()

    def training_epoch_end(self, outputs):
        # here, we have to put push_prototypes function
        # logs costs after a training epoch
        if self.current_epoch >= self.push_start and self.current_epoch in self.push_epochs:

            self.update_prototypes(self.trainer.train_dataloader.loaders)

            val_cost = self.custom_validation_epoch()
            # TODO: save the model if the performance metric is better

            self.last_only()
            with tqdm(
                total=10,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [train_cost={postfix[0][train_cost]}'
                ' val_cost={postfix[1][val_cost]}]',
                desc='Training last only',
                postfix=[dict(train_cost=0), dict(val_cost=0)],
            ) as t:
                for i in range(10):
                    train_cost = self.custom_train_epoch(t)
                    val_cost = self.custom_validation_epoch(t)
                    t.update()

                # save model performance
                # TODO: calculate performance and update the global performance criterium, if it is
                # worse

                # optionally (plot something)

    def validation_step(self, batch, batch_idx):
        cost, f1_score = self.train_val_test(batch)
        self.log('val_cost', cost, prog_bar=True)
        self.log('f1_score_val', f1_score, prog_bar=True)
        return cost

    def test_step(self, batch, batch_idx):
        cost, f1_score = self.train_val_test(batch)
        return cost

    def custom_validation_epoch(self, t: Optional[tqdm] = None) -> torch.Tensor:
        if self.training:
            self.eval()
        for batch in self.trainer.val_dataloaders[0]:
            with torch.no_grad():
                val_cost, f1_score = self.train_val_test(batch)
            if t:
                # this implementation implies that only the last cost will be saved and shown
                t.postfix[1]['val_cost'] = round(val_cost.item(), 2)
        return val_cost

    def custom_train_epoch(self, t: Optional[tqdm] = None) -> torch.Tensor:
        if not self.training:
            self.train()
        for batch in self.trainer.train_dataloader.loaders:
            with torch.enable_grad():
                train_cost, f1_score = self.train_val_test(batch)
            if t:
                # this implementation implies that only the last cost will be saved and shown
                t.postfix[0]['train_cost'] = round(train_cost.item(), 2)
            train_cost.backward()
            self.trainer.optimizers[0].step()
            self.trainer.optimizers[0].zero_grad()
        return train_cost

    def train_val_test(self, batch):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        # labels - (batch, 7), dtype: float32
        # images - (batch, 10, H, W)
        # images have to have the shape (batch, 10, H, W). 7 extra channel implies fine annotations,
        # in case of 7 classes
        if images.shape[1] != 12:
            raise Exception("The channel dimension of the input-image batch has to be 10")

        fine_annotations = images[:, 3:12, :, :]  # 9 classes of fine annotations
        images = images[:, 0:3, :, :]  # (no view, create slice)

        # images = images.cuda()
        # labels = labels.cuda()
        # fine_annotations = fine_annotations.cuda()

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

        cost = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 0.001 * fine_cost

        f1_score = multilabel_f1_score(output, labels, num_labels=self.num_classes, threshold=0.5)

        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        return cost, f1_score

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

    def compute_layer_rf_info(
        self, layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info
    ):
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
            n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1

        pL = math.floor(pad / 2)

        j_out = j_in * layer_stride
        r_out = r_in + (layer_filter_size - 1) * j_in
        start_out = start_in + ((layer_filter_size - 1) / 2 - pL) * j_in

        return [n_out, j_out, r_out, start_out]

    def compute_proto_layer_rf_info(
        self, img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size
    ):
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

    def compute_rf_protoL_at_spatial_location(
        self, img_size, height_index, width_index, protoL_rf_info
    ):
        # computes the pixel indices of the input-image patch (e.g. 224x224) that corresponds
        # to the feature-map patch with the closest distance to the current prototype
        n = protoL_rf_info[0]
        j = protoL_rf_info[1]
        r = protoL_rf_info[2]
        start = protoL_rf_info[3]
        assert height_index < n
        assert width_index < n

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

    def compute_rf_prototype(self, img_size, prototype_patch_index, protoL_rf_info):
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = self.compute_rf_protoL_at_spatial_location(
            img_size, height_index, width_index, protoL_rf_info
        )
        return [img_index, rf_indices[0], rf_indices[1], rf_indices[2], rf_indices[3]]

    def find_high_activation_crop(self, activation_map, percentile=95):
        threshold = np.percentile(activation_map, percentile)
        mask = np.ones(activation_map.shape)
        mask[activation_map < threshold] = 0
        lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
        for i in range(mask.shape[0]):
            if np.amax(mask[i]) > 0.5:
                lower_y = i
                break
        for i in reversed(range(mask.shape[0])):
            if np.amax(mask[i]) > 0.5:
                upper_y = i
                break
        for j in range(mask.shape[1]):
            if np.amax(mask[:, j]) > 0.5:
                lower_x = j
                break
        for j in reversed(range(mask.shape[1])):
            if np.amax(mask[:, j]) > 0.5:
                upper_x = j
                break
        return lower_y, upper_y + 1, lower_x, upper_x + 1

    def update_prototypes(
        self,
        dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes='./savings/',  # if not None, prototypes will be saved here
        prototype_img_filename_prefix='prototype-img',
        prototype_self_act_filename_prefix='prototype-self-act',
        proto_bound_boxes_filename_prefix='bb',
        # log=print,
    ):
        self.eval()
        prototype_shape = self.prototype_shape
        n_prototypes = self.num_prototypes
        # train dataloader
        # dataloader = dataloader  # self.trainer.train_dataloader.loaders

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
        # 5: class identities
        # proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        # proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_rf_boxes = {}
        proto_bound_boxes = {}

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

        for push_iter, batch in enumerate(dataloader):
            search_batch_images, search_labels = batch
            if search_batch_images.shape[1] > 3:
                # only imagees (the extra channels in this dimension belong to fine annot masks)
                search_batch_images = search_batch_images[:, 0:3, :, :]

            start_index_of_search_batch = push_iter * search_batch_size

            self.update_prototypes_on_batch(
                search_batch_images,
                start_index_of_search_batch,
                search_labels,
                global_min_proto_dist,
                global_min_fmap_patches,
                proto_rf_boxes,
                proto_bound_boxes,
                prototype_layer_stride=prototype_layer_stride,
                dir_for_saving_prototypes=proto_epoch_dir,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            )

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
                    search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info
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
                proto_bound_boxes[j]['class_indentities'] = search_labels[
                    rf_prototype_j[0]
                ].tolist()

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
                            np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET
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

        del class_to_img_index_dict


if __name__ == "__main__":

    torch.cuda.empty_cache()

    sq_net = SqueezeNet()

    edema_net_st = EdemaNet(sq_net, 9, prototype_shape=(9, 512, 1, 1), img_size=300)
    edema_net = edema_net_st.cuda()

    images = torch.rand(32, 3, 300, 300)
    images = images.cuda()
    x = edema_net.encoder(images)
    print(x.shape)
    x = edema_net.transient_layers(x)
    print(x.shape)

    print(edema_net.proto_layer_rf_info)
    # logits, _, _ = edema_net.forward(images)
    # prop = torch.nn.functional.softmax(logits, dim=1)

    # test_dataset = TensorDataset(
    #     torch.rand(128, 10, 400, 400), torch.randint(0, 2, (128, 7), dtype=torch.float32)
    # )
    # test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # trainer = pl.Trainer(max_epochs=9, logger=False, enable_checkpointing=False, gpus=1)
    # trainer.fit(edema_net, test_dataloader)
