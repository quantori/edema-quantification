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
        num_warm_epochs: int = 10,
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
        # cross entropy cost function
        self.cross_entropy_cost = torch.nn.BCEWithLogitsLoss()

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

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.num_warm_epochs:
            pass

        else:
            pass

            if self.epoch >= push_start and self.epoch in push_epochs:
                pass

    def train_func(self, batch):
        images, labels = batch
        # labels - (batch, 7)
        # images - (batch, 10, H, W)
        # images have to have the shape (batch, 10, H, W). 7 extra channel implies fine annotations,
        # in case of 7 classes
        if images.shape[1] != 10:
            raise Exception("The channel dimension of the input-image batch has to be 10")

        fine_annotations = images[:, 3:10, :, :]  # 7 classes of fine annotations
        images = images[:, 0:3, :, :]  # (no view, create slice)

        # nn.Module has implemented __call__() function, so no need to call .forward()
        output, min_distances, upsampled_activation = self(images)

        # classification cost
        cross_entropy = self.cross_entropy_cost(output, labels)

        # cluster cost
        max_dist = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
        prototypes_of_correct_class = torch.matmul(
            labels, torch.permute(self.prototype_class_identity, (1, 0))
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

    def test_func():
        pass

    def warm_only(model):
        for p in model.module.encoder.parameters():
            p.requires_grad = False
        for p in model.module.transient_layers.parameters():
            p.requires_grad = True
            model.module.prototype_layer.requires_grad = True
        for p in model.module.last_layer.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        pass
        # return torch.optim.Adam(self.parameters(), lr=0.02)

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

    def update_prototypes_on_batch():
        pass


if __name__ == "__main__":

    sq_net = SqueezeNet()
    # summary(sq_net.model, (3, 224, 224))
    edema_net = EdemaNet(sq_net, 7, prototype_shape=(35, 512, 1, 1))

    search_y = torch.randint(0, 2, (64,))
    print(search_y)
    distances = torch.rand(64, 35, 14, 14)
    convs = torch.rand(64, 512, 14, 14)
    proto_dist_j = torch.rand(1, 1, 14, 14)
    num_classes = 7
    proto_dist_j = distances[[0, 5, 10, 15, 25]][:, 1, :, :]
    print(proto_dist_j.shape)
    class_to_img_index_dict = {1: [0, 5, 10, 15, 25]}
    # print(proto_dist_j)
    # na = proto_dist_j.numpy()
    # print(na.shape)
    batch_min_proto_dist_j = torch.amin(proto_dist_j)
    # batch_min_proto_dist_np = np.amin(na)
    print(batch_min_proto_dist_j)

    batch_argmin_proto_dist_j = list(
        np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
    )
    print(batch_argmin_proto_dist_j)

    batch_argmin_proto_dist_j[0] = class_to_img_index_dict[1][batch_argmin_proto_dist_j[0]]
    print(batch_argmin_proto_dist_j)

    # print(batch_min_proto_dist_j)
    # print(batch_min_proto_dist_np)

    # class_to_img_index_dict = {key: [] for key in range(num_classes)}
    # # img_y is the image's integer label
    # for img_index, img_y in enumerate(search_y):
    #     img_label = img_y.item()
    #     class_to_img_index_dict[img_label].append(img_index)

    # prototype_shape = prototype_network_parallel.module.prototype_shape
    # n_prototypes = prototype_shape[0]
    # proto_h = prototype_shape[2]
    # proto_w = prototype_shape[3]
    # max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    # for j in range(n_prototypes):
    #     #if n_prototypes_per_class != None:
    #     if class_specific:
    #         # target_class is the class of the class_specific prototype
    #         target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
    #         # if there is not images of the target_class from this batch
    #         # we go on to the next prototype
    #         if len(class_to_img_index_dict[target_class]) == 0:
    #             continue
    #         proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
    #     else:
    #         # if it is not class specific, then we will search through
    #         # every example
    #         # proto_dist_j = proto_dist_[:,j,:,:]
    #         target_class = 1
    #         # if there is not images of the target_class from this batch
    #         # we go on to the next prototype
    #         if len(class_to_img_index_dict[target_class]) == 0:
    #             continue
    #         proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]

    #     batch_min_proto_dist_j = np.amin(proto_dist_j)
    #     if batch_min_proto_dist_j < global_min_proto_dist[j]:
    #         batch_argmin_proto_dist_j = \
    #             list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
    #                                   proto_dist_j.shape))