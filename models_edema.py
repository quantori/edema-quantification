"""Models for the edema classification project.

The description to be filled...
"""

from turtle import shape
from typing import Tuple
from matplotlib import image
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np
import torch.nn.functional as F


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

        self.model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", weights="DEFAULT")
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

        # onehot indication matrix for prototypes (num_prototypes, num_classes)
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        num_prototypes_per_class = self.num_prototypes // num_classes
        # fills with 1 only those prototypes, which correspond to the correct class. The rest is
        # filled with 0
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # encoder
        self.encoder = encoder

        # transient layers
        self.transient_layers = self._make_transient_layers(self.encoder)

        # prototypes layer (do not make this just a tensor, since it will not be moved
        # automatically to gpu)
        self.prototype_layer = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # last fully connected layer for the classification of edema features. The bias is not used
        # in the original paper
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)

    def forward(self, x):
        # x is of dimension (batch, 4, spatial, spatial). The one extra channel is fine annotations
        x = x[:, 0:3, :, :]  # (no view; create slice. When no fa is available this will return x)

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
            size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )(activation)

        return logits, min_distances, upsampled_activation

    def training_step(self, batch, batch_idx):

        # retrieve fine annotations
        images, labels = batch
        if self.fine_loader:
            fine_images, fine_labels = next(iter(self.fine_loader))
            # print(image.shape)
            # image and  fine_image have to have the same channel dimension (batch,  channel, W, H)
            images = torch.cat((images, fine_images))
            labels = torch.cat((labels, fine_labels))
            # print(image.shape)
        if images.shape[1] == 4:
            with_fa = True
            fine_annotations = image[:, 3:4, :, :]
            images = images[:, 0:3, :, :]  # (no view, create slice)
        elif images.shape[1] == 3:
            # means everything can be relevant
            fine_annotations = torch.zeros(size=(image.shape[0], 1, image.shape[2], image.shape[3]))
            images = images
        fine_annotations = fine_annotations.cuda()
        input = images.cuda()
        target = labels.cuda()

        # nn.Module has implemented __call__() function, so no need to call .forward()
        output, min_distances, upsampled_activation = self(input)

        # classification loss
        cross_entropy = torch.nn.functional.cross_entropy(output, target)

        # cluster cost
        max_dist = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        cluster_cost = self.cluster_cost(
            max_dist, min_distances, self.prototype_class_identity, labels
        )

        # separation cost
        separation_cost = self.separation_cost(
            max_dist, min_distances, self.prototype_class_identity, labels
        )

        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # return loss

    def configure_optimizers(self):
        pass
        # return torch.optim.Adam(self.parameters(), lr=0.02)

    def prototype_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Returns prototype distances.

        Slices through the image in the latent space and calculates a p-norm distance between the
        image patches and prototypes.

        Args:
            x (torch.Tensor): image after convolution layers.
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
        max_dist: int,
        min_distances: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cluster cost.

        Args:
            max_dist (int): max distance is needed for the inverting trick (have a look below)
            min_distances (torch.Tensor): min distances returned by the model.forward()
            prototype_class_identity (torch.Tensor): onehot prototype indication matrix
            labels (torch.Tensor): batch of labels taken from a dataloader

        Returns:
            torch.Tensor: cluster cost, which makes prototypes of the same class closer to each
                          other
        """

        # prototypes_of_correct_class is a tensor of shape (batch_size, num_prototypes)
        prototypes_of_correct_class = torch.t(prototype_class_identity[:, labels]).cuda()
        # (max_dist - min_distances) and, consequently, inverted_distances is done to prevent the
        # constant retrieving of 0 for min values obtained by *prototypes_of_correct_class
        inverted_distances, _ = torch.max(
            (max_dist - min_distances) * prototypes_of_correct_class, dim=1
        )
        cluster_cost = torch.mean(max_dist - inverted_distances)

        return cluster_cost

    def separation_cost(
        max_dist: int,
        min_distances: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Separation cost.

        Args:
            max_dist (int): max distance is needed for the inverting trick (have a look below)
            min_distances (torch.Tensor): min distances returned by the model.forward()
            prototype_class_identity (torch.Tensor): onehot prototype indication matrix
            labels (torch.Tensor): batch of labels taken from a dataloader

        Returns:
            torch.Tensor: separation cost, which makes prototypes of different classes farer from
                          each other
        """

        # prototypes_of_correct_class is a tensor of shape (batch_size, num_prototypes)
        prototypes_of_correct_class = torch.t(prototype_class_identity[:, labels]).cuda()
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        # (max_dist - min_distances) and, consequently, inverted_distances is done to prevent the
        # constant retrieving of 0 for min values obtained by *prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = torch.max(
            (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
        )
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        return separation_cost

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
            print(first_transient_layer_in_channels)
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


if __name__ == "__main__":

    sq_net = SqueezeNet()
    # summary(sq_net.model, (3, 224, 224))
    edema_net = EdemaNet(sq_net, 3, prototype_shape=(15, 512, 1, 1))
    # print(edema_net._make_transient_layers(sq_net.model))
    convoluted_image = torch.rand(1, 512, 14, 14)
    images = torch.rand(64, 3, 300, 300)
    labels = torch.randint(0, 3, (64,))
    fine_images = torch.rand(10, 3, 300, 300)

    print(labels)

    _, min_distances, _ = edema_net(images)

    print(min_distances.shape)

    # print(torch.cat((y, z)).shape)

    # y = edema_net.encoder(y)
    # y = edema_net.transient_layers(y)
    # distances = edema_net.prototype_distances(y)
    # print(distances.shape)

    # _distances = distances.view(distances.shape[0], distances.shape[1], -1)
    # # print(_distances)
    # closest_k_distances, _ = torch.topk(_distances, 5, largest=False)
    # # print(closest_k_distances)
    # min_distances = F.avg_pool1d(
    #     closest_k_distances, kernel_size=closest_k_distances.shape[2]
    # ).view(-1, 5)
    # print(min_distances.shape)

    num_prototypes = 15
    num_classes = 3
    num_prototypes_per_class = num_prototypes // num_classes
    max_dist = 512

    prototype_class_identity = torch.zeros(num_prototypes, num_classes)
    # print(prototype_class_identity)

    for j in range(num_prototypes):
        prototype_class_identity[j, j // num_prototypes_per_class] = 1
    # print(prototype_class_identity)

    prototypes_of_correct_class = torch.t(prototype_class_identity[:, labels])
    print(prototypes_of_correct_class.shape)

    correct_distances = (max_dist - min_distances) * prototypes_of_correct_class
    print(correct_distances)

    inverted_distances, _ = torch.max(
        (max_dist - min_distances) * prototypes_of_correct_class, dim=1
    )
    print(inverted_distances)

    print(max_dist - inverted_distances)

    cluster_cost = torch.mean(max_dist - inverted_distances)
    print(cluster_cost)

    # print(edema_net.forward(y)[0].shape)

    # prototype_activations = torch.log((distances + 1) / (distances + 1e-4))
    # print(prototype_activations.shape)
    # _activations = prototype_activations.view(
    #     prototype_activations.shape[0], prototype_activations.shape[1], -1
    # )
    # print(_activations.shape)
    # top_k_activations, _ = torch.topk(_activations, 5)
    # print(top_k_activations.shape)
    # prototype_activations = F.avg_pool1d(
    #     top_k_activations, kernel_size=top_k_activations.shape[2]
    # ).view(-1, 5)
    # print(prototype_activations.shape)
    # logits = edema_net.last_layer(prototype_activations)
    # print(logits)
    # logits[:, 0] = 0
    # print(logits)
    # # print(x[0][0])
    # # print(edema_net.prototype_layer[0][0])
    # y = edema_net.encoder(y)
    # print(y.shape)
    # y = edema_net.transient_layers(y)
    # print(y.shape)
    # distances = edema_net.prototype_distances(y)
    # print(distances.shape)
    # print(sq_net.model.__class__.__name__)
    # for module in sq_net.model.modules():
    #     print(module)
    # print(dict(sq_net.model.named_children()))
    # print(sq_net.model.children()[0])

    # y1 = torch.rand(1, 196, 512)
    # y2 = torch.rand(1, 1, 512)
    # d = torch.cdist(y1, y2)
    # print(d)
