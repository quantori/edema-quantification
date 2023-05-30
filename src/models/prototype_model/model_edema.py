"""Model for the edema classification project.

The description to be filled...
"""
import os
import sys
from statistics import mean
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from encoders import IEncoderEdema
from last_layers import ILastLayers
from loggers import IPrototypeLogger
from omegaconf import DictConfig
from prototype_layers import IPrototypeLayer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_f1_score
from tqdm.auto import tqdm
from transient_layers import ITransientLayers


class EdemaPrototypeNet(pl.LightningModule):
    """Prototype edema model class.

    A complete model is implemented (includes the encoder, transient, prototype and fully connected
    layers). The transient layers are required to concatenate the main encoder layers with the
    prototype layer. The encoder is the variable part of EdemaNet, which is passed as an argument.

    Args:
        encoder: the encoder of the EdemaNet.
        transient_layers: the transient of the EdemaNet.
        prototype_layer: the prototype layer of the EdemaNet.
        last_layer: the last layer of the EdemaNet.
        settings_model: the model's settings from a hydra config.
        prototype_logger: a custom logger for logging training artefacts.
    """

    def __init__(
        self,
        encoder: IEncoderEdema,
        transient_layers: ITransientLayers,
        prototype_layer: IPrototypeLayer,
        last_layer: ILastLayers,
        settings_model: DictConfig,
        prototype_logger: Optional[IPrototypeLogger] = None,
    ) -> None:
        super().__init__()
        self.top_k = settings_model.top_k  # for a 14x14: top_k=3 is 1.5%, top_k=9 is 4.5%
        self.epsilon = settings_model.epsilon  # needed for the similarity calculation
        self.num_classes = settings_model.num_classes
        self.num_warm_epochs = settings_model.num_warm_epochs
        self.push_start = settings_model.push_start
        self.push_epochs = settings_model.push_epochs

        if prototype_logger is not None:
            self._prototype_logger = prototype_logger

        # cross entropy cost function
        self.cross_entropy_cost = nn.BCEWithLogitsLoss()

        # encoder
        self.encoder = encoder

        # transient layers
        self.transient_layers = transient_layers

        # prototypes layer (do not make this just a tensor, since it will not be moved
        # automatically to gpu)
        self.prototype_layer = prototype_layer

        # last fully connected layer for the classification of edema features. The bias is not used
        # in the original paper
        self.last_layer = last_layer

        self.blocks = {
            'encoder': self.encoder,
            'transient_layers': self.transient_layers,
            'prototype_layer': self.prototype_layer,
            'last_layer': self.last_layer,
        }

        # Crutches for a custom training loop
        self._training_status = ' '
        self._real_epoch = 0
        self._num_last_epochs = settings_model.num_last_epochs
        self._last_step = settings_model.num_last_epochs

    def forward(self, x):
        # x is a batch of images having (batch, 3, H, W) dimensions
        if x.shape[1] != 3:
            raise Exception('The channel dimension of the input-image batch has to be 3')

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
            closest_k_distances,
            kernel_size=closest_k_distances.shape[2],
        ).view(-1, self.prototype_layer.num_prototypes)

        prototype_activations = torch.log((distances + 1) / (distances + self.epsilon))
        _activations = prototype_activations.view(
            prototype_activations.shape[0],
            prototype_activations.shape[1],
            -1,
        )
        top_k_activations, _ = torch.topk(_activations, self.top_k)
        prototype_activations = F.avg_pool1d(
            top_k_activations,
            kernel_size=top_k_activations.shape[2],
        ).view(-1, self.prototype_layer.num_prototypes)

        logits = self.last_layer(prototype_activations)

        activation = torch.log((distances + 1) / (distances + self.epsilon))
        upsampled_activation = torch.nn.Upsample(
            size=(upsample_hight, upsample_width),
            mode='bilinear',
            align_corners=False,
        )(activation)

        return logits, min_distances, upsampled_activation

    def update_prototypes_forward(self, x):
        conv_output = self.encoder(x)
        conv_output = self.transient_layers(conv_output)
        distances = self.prototype_distances(conv_output)
        return conv_output, distances

    def training_step(self, batch, batch_idx):
        # based on universal train_val_test(). Logs costs after each train step
        loss, f1_train = self.train_val_test(batch)
        self.log('f1_train', f1_train, prog_bar=True)
        self.log('train_loss', loss)
        return {'loss': loss, 'f1_train': f1_train}

    def on_train_epoch_start(self):
        if self._real_epoch < self.num_warm_epochs:
            _warm(**self.blocks)
            _print_status_bar(self.trainer, self.blocks, status='WARM')
            self._real_epoch += 1
        elif self._training_status == 'last':
            _last(**self.blocks)
            self._last_step -= 1
            _print_status_bar(
                self.trainer,
                self.blocks,
                status=f'LAST-({self._last_step} epochs left)',
            )
        else:
            _joint(**self.blocks)
            _print_status_bar(self.trainer, self.blocks, status='JOINT')
            self._real_epoch += 1

    def on_train_epoch_end(self):
        # here, we have to put push_prototypes function
        # logs costs after a training epoch
        if self._real_epoch >= self.push_start and self._real_epoch in self.push_epochs:
            if self._last_step == self._num_last_epochs:
                self.prototype_layer.update(
                    self,
                    self.trainer.train_dataloader.loaders,
                    self._prototype_logger,
                )
                self._training_status = 'last'
                self.train()
            elif self._last_step == 0:
                self._training_status = 'joint'
                self._last_step = self._num_last_epochs
            # self.val_epoch(self.trainer.val_dataloaders[0], position=3)
            # self.trainer.validate(self, self.trainer.train_dataloader.loaders)
            # TODO: save the model if the performance metric is better
            # self.train_last_only(
            #     self.trainer.train_dataloader.loaders,
            #     self.trainer.val_dataloaders[0],
            # )

            # save model performance
            # TODO: calculate performance and update the global performance criterium, if it is
            # worse

    def validation_step(self, batch, batch_idx):
        loss, f1_val = self.train_val_test(batch)
        self.log('f1_val', f1_val, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss)
        return {'loss': loss, 'f1_val': f1_val}

    def test_step(self, batch, batch_idx):
        cost, f1_score = self.train_val_test(batch)
        return cost

    def custom_val_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss, f1_val = self.train_val_test(batch)
        self.log('val_loss', loss)
        return {'loss': loss, 'f1_val': f1_val}

    def val_epoch(
        self,
        dataloader: DataLoader,
        t: Optional[tqdm] = None,
        position: int = 4,
    ) -> None:
        f1_all = []
        self.eval()
        with torch.no_grad():
            with tqdm(
                total=len(dataloader),
                desc='Validating',
                position=position,
                leave=False,
            ) as t1:
                for idx, batch in enumerate(dataloader):
                    preds = self.custom_val_step(batch)
                    f1_all.append(preds['f1_val'].item())
                    t1.update()
                if t:
                    preds = self.refactor_val_dict(preds)
                    to_postfix = self.str_to_dict(t.postfix)
                    to_postfix.update({'f1_val': round(preds['f1_val'], 3)})
                    t.set_postfix(to_postfix)
                    t.n += idx + 1
                    t.refresh()
        if mean(f1_all) > self.trainer.logged_metrics['f1_val']:
            _save_new_checkpoint(
                self.trainer.checkpoint_callback.best_model_path,
                self.trainer,
                mean(f1_all),
            )
        self.log('f1_val', mean(f1_all), on_step=False, prog_bar=True)
        self.train()

    def train_epoch(self, dataloader: DataLoader, t: tqdm):
        for idx, batch in enumerate(dataloader):
            preds = self.training_step(batch, idx)
            preds = self.refactor_train_dict(preds)
            if t.postfix is not None:
                to_postfix = self.str_to_dict(t.postfix)
                to_postfix.update(preds)
                t.set_postfix(to_postfix)
            else:
                t.set_postfix(preds)
            t.n = idx + 1
            t.refresh()

    def train_last_only(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 2,
    ) -> None:
        _last(**self.blocks)
        _print_status_bar(self.trainer, self.blocks, status='LAST')
        with tqdm(
            leave=False,
            dynamic_ncols=True,
            position=3,
            file=sys.stdout,
        ) as t:
            total = len(train_dataloader) + len(val_dataloader)
            for epoch in range(epochs):
                t.reset(total=total)
                t.initial = 0
                t.set_description(f'Training last, Epoch {epoch}')
                self.train_epoch(train_dataloader, t)
                self.val_epoch(val_dataloader, t)
            t.close()

    def refactor_train_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
        d_new = {}
        d_new['loss_train'] = d['loss'].item()
        d_new['f1_train'] = d['f1_train'].item()
        del d
        return d_new

    def refactor_val_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
        d_new = {}
        d_new['loss_val'] = d['loss'].item()
        d_new['f1_val'] = d['f1_val'].item()
        del d
        return d_new

    def str_to_dict(self, string: str) -> Dict[str, float]:
        # pattern for string -- 'f1_train=0.001, loss_train=0.001, ...'
        d = {}
        splitted_str = string.split(',')
        for sub in splitted_str:
            sub = sub.strip()
            sub_list = sub.split('=')
            d.update({sub_list[0]: float(sub_list[1])})
        return d

    def train_val_test(self, batch):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        # labels - (batch, 7), dtype: float32
        # images - (batch, 10, H, W)
        # images have to have the shape (batch, 10, H, W). 7 extra channel implies fine annotations,
        # in case of 7 classes
        if images.shape[1] != 12:
            raise Exception('The channel dimension of the input-image batch has to be 10')

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
        max_dist = (
            self.prototype_layer.shape[1]
            * self.prototype_layer.shape[2]
            * self.prototype_layer.shape[3]
        )
        prototypes_of_correct_class = torch.matmul(
            labels,
            self.prototype_layer.prototype_class_identity.permute(1, 0),
        )
        cluster_cost = self.cluster_cost(max_dist, min_distances, prototypes_of_correct_class)

        # separation cost
        separation_cost = self.separation_cost(max_dist, min_distances, prototypes_of_correct_class)

        # fine cost
        fine_cost = self.fine_cost(
            fine_annotations,
            upsampled_activation,
            self.prototype_layer.num_prototypes_per_class,
        )

        cost = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 0.001 * fine_cost

        f1_score = multilabel_f1_score(output, labels, num_labels=self.num_classes, threshold=0.3)

        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        return cost, f1_score

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
        expanded_x = nn.Unfold(
            kernel_size=(self.prototype_layer.shape[2], self.prototype_layer.shape[3]),
        )(x)

        # expanded shape = [1, batch, number of such blocks, channel*proto_shape[2]*proto_shape[3]]
        expanded_x = expanded_x.unsqueeze(0).permute(0, 1, 3, 2)

        # change the input tensor into contiguous in memory tensor (make a copy). The output of the
        # view() (if input=(1, batch, 196, 512)) -> a tensor of (1, batch*196, 512=512*1*1)
        # dimension (if the prototype dimensions are (num_prototypes, 512, 1, 1)). -1 means that
        # this dimension is calculated based on other dimensions
        expanded_x = expanded_x.contiguous().view(
            1,
            -1,
            self.prototype_layer.shape[1]
            * self.prototype_layer.shape[2]
            * self.prototype_layer.shape[3],
        )

        # if the input tensor is (num_prototypes, 512, 1, 1) and the kernel_size=(1, 1), the output
        # is (1, num_prototypes, 512=512*1*1, 1=1*1).
        # Expanded proto shape = [1, proto num, channel*proto_shape[2]*proto_shape[3], 1]
        expanded_proto = nn.Unfold(
            kernel_size=(self.prototype_layer.shape[2], self.prototype_layer.shape[3]),
        )(self.prototype_layer).unsqueeze(0)

        # if the input is (1, num_prototypes, 512, 1), the output is (1, num_prototypes, 512=512*1)
        expanded_proto = expanded_proto.contiguous().view(1, expanded_proto.shape[1], -1)

        # the output is (1, Batch * number of blocks in x, num_prototypes). If expanded_x is
        # (1, batch*196, 512) and expanded_proto is (1, num_prototypes, 512), the output shape is
        # (1, batch*196, num_prototypes). The default p value for the p-norm distance = 2 (euclidean
        # distance)
        expanded_distances = torch.cdist(expanded_x, expanded_proto)

        # (1, batch*196, num_prototypes) -> (batch, num_prototypes, 1*196)
        expanded_distances = torch.reshape(
            expanded_distances,
            shape=(batch, -1, self.prototype_layer.shape[0]),
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
                self.prototype_layer.shape[0],
                x.shape[2] - self.prototype_layer.shape[2] + 1,
                x.shape[3] - self.prototype_layer.shape[3] + 1,
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
            (max_dist - min_distances) * prototypes_of_correct_class,
            dim=1,
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
            (max_dist - min_distances) * prototypes_of_wrong_class,
            dim=1,
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

    def compute_rf_protoL_at_spatial_location(
        self,
        img_size,
        height_index,
        width_index,
        protoL_rf_info,
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

    def compute_rf_prototype(self, img_size, prototype_patch_index, protoL_rf_info):
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = self.compute_rf_protoL_at_spatial_location(
            img_size,
            height_index,
            width_index,
            protoL_rf_info,
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

    # def update_prototypes(
    #     self,
    #     dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    #     prototype_layer_stride=1,
    #     root_dir_for_saving_prototypes='./savings/',  # if not None, prototypes will be saved here
    #     prototype_img_filename_prefix='prototype-img',
    #     prototype_self_act_filename_prefix='prototype-self-act',
    #     proto_bound_boxes_filename_prefix='bb',
    #     # log=print,
    # ):
    #     self.eval()
    #     prototype_shape = self.prototype_shape
    #     n_prototypes = self.num_prototypes
    #     # train dataloader
    #     # dataloader = dataloader  # self.trainer.train_dataloader.loaders

    #     # make an array for the global closest distance seen so far (initialized with floating point
    #     # representation of positive infinity)
    #     global_min_proto_dist = np.full(n_prototypes, np.inf)

    #     # saves the patch representation that gives the current smallest distance
    #     global_min_fmap_patches = np.zeros(
    #         [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]],
    #     )

    #     # proto_rf_boxes (receptive field) and proto_bound_boxes column:
    #     # 0: image index in the entire dataset
    #     # 1: height start index
    #     # 2: height end index
    #     # 3: width start index
    #     # 4: width end index
    #     # 5: class identities
    #     # proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    #     # proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    #     proto_rf_boxes = {}
    #     proto_bound_boxes = {}

    #     # making a directory for saving prototypes
    #     if root_dir_for_saving_prototypes != None:
    #         if self.current_epoch != None:
    #             proto_epoch_dir = os.path.join(
    #                 root_dir_for_saving_prototypes,
    #                 'epoch-' + str(self.current_epoch),
    #             )
    #             if not os.path.exists(proto_epoch_dir):
    #                 os.makedirs(proto_epoch_dir)
    #         else:
    #             proto_epoch_dir = root_dir_for_saving_prototypes
    #     else:
    #         proto_epoch_dir = None

    #     search_batch_size = dataloader.batch_size

    #     with tqdm(total=len(dataloader), desc='Updating prototypes', position=3, leave=False) as t:
    #         for push_iter, batch in enumerate(dataloader):
    #             search_batch_images, search_labels = batch
    #             if search_batch_images.shape[1] > 3:
    #                 # only imagees (the extra channels in this dimension belong to fine annot masks)
    #                 search_batch_images = search_batch_images[:, 0:3, :, :]

    #             start_index_of_search_batch = push_iter * search_batch_size

    #             self.update_prototypes_on_batch(
    #                 search_batch_images,
    #                 start_index_of_search_batch,
    #                 search_labels,
    #                 global_min_proto_dist,
    #                 global_min_fmap_patches,
    #                 proto_rf_boxes,
    #                 proto_bound_boxes,
    #                 prototype_layer_stride=prototype_layer_stride,
    #                 dir_for_saving_prototypes=proto_epoch_dir,
    #                 prototype_img_filename_prefix=prototype_img_filename_prefix,
    #                 prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
    #             )
    #             t.update()

    #     if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
    #         proto_rf_boxes_json = json.dumps(proto_rf_boxes)
    #         f = open(
    #             os.path.join(
    #                 proto_epoch_dir,
    #                 proto_bound_boxes_filename_prefix
    #                 + '-receptive_field'
    #                 + str(self.current_epoch)
    #                 + '.json',
    #             ),
    #             'w',
    #         )
    #         f.write(proto_rf_boxes_json)
    #         f.close()

    #         proto_bound_boxes_json = json.dumps(proto_bound_boxes)
    #         f = open(
    #             os.path.join(
    #                 proto_epoch_dir,
    #                 proto_bound_boxes_filename_prefix + str(self.current_epoch) + '.json',
    #             ),
    #             'w',
    #         )
    #         f.write(proto_bound_boxes_json)
    #         f.close()

    #     prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    #     self.prototype_layer.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    #     # # prototype_network_parallel.cuda()
    #     # end = time.time()
    #     # log('\tpush time: \t{0}'.format(end - start))

    # def update_prototypes_on_batch(
    #     self,
    #     search_batch_images,
    #     start_index_of_search_batch,
    #     search_labels,
    #     global_min_proto_dist,  # this will be updated
    #     global_min_fmap_patches,  # this will be updated
    #     proto_rf_boxes,  # this will be updated
    #     proto_bound_boxes,  # this will be updated
    #     prototype_layer_stride=1,
    #     dir_for_saving_prototypes=None,
    #     prototype_img_filename_prefix=None,
    #     prototype_self_act_filename_prefix=None,
    # ):
    #     # Model has to be in the eval mode
    #     if self.training:
    #         self.eval()

    #     with torch.no_grad():
    #         search_batch = search_batch_images.cuda()
    #         # this computation currently is not parallelized
    #         protoL_input_torch, proto_dist_torch = self.update_prototypes_forward(search_batch)

    #     protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    #     proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    #     del protoL_input_torch, proto_dist_torch

    #     # form a dict with {class:[images_idxs]}
    #     class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
    #     for img_index, img_y in enumerate(search_labels):
    #         img_y.tolist()
    #         for idx, i in enumerate(img_y):
    #             if i:
    #                 class_to_img_index_dict[idx].append(img_index)

    #     prototype_shape = self.prototype_shape
    #     n_prototypes = prototype_shape[0]
    #     proto_h = prototype_shape[2]
    #     proto_w = prototype_shape[3]
    #     # max_dist is chosen arbitrarly
    #     max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    #     for j in range(n_prototypes):
    #         # target_class is the class of the class_specific prototype
    #         target_class = torch.argmax(self.prototype_class_identity[j]).item()
    #         # if there is not images of the target_class from this batch
    #         # we go on to the next prototype
    #         if len(class_to_img_index_dict[target_class]) == 0:
    #             continue
    #         proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]

    #         # if the smallest distance in the batch is less than the global smallest distance for
    #         # this prototype
    #         batch_min_proto_dist_j = np.amin(proto_dist_j)
    #         if batch_min_proto_dist_j < global_min_proto_dist[j]:
    #             # find arguments of the smallest distance in the matrix shape
    #             arg_min_flat = np.argmin(proto_dist_j)
    #             arg_min_matrix = np.unravel_index(arg_min_flat, proto_dist_j.shape)
    #             batch_argmin_proto_dist_j = list(arg_min_matrix)

    #             # change the index of the smallest distance from the class specific index to the
    #             # whole search batch index
    #             batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][
    #                 batch_argmin_proto_dist_j[0]
    #             ]

    #             # retrieve the corresponding feature map patch
    #             img_index_in_batch = batch_argmin_proto_dist_j[0]
    #             fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
    #             fmap_height_end_index = fmap_height_start_index + proto_h
    #             fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
    #             fmap_width_end_index = fmap_width_start_index + proto_w

    #             batch_min_fmap_patch_j = protoL_input_[
    #                 img_index_in_batch,
    #                 :,
    #                 fmap_height_start_index:fmap_height_end_index,
    #                 fmap_width_start_index:fmap_width_end_index,
    #             ]

    #             global_min_proto_dist[j] = batch_min_proto_dist_j
    #             global_min_fmap_patches[j] = batch_min_fmap_patch_j

    #             # get the receptive field boundary of the image patch (the input image e.g. 224x224)
    #             # that generates the representation
    #             protoL_rf_info = self.proto_layer_rf_info
    #             rf_prototype_j = self.compute_rf_prototype(
    #                 search_batch.size(2),
    #                 batch_argmin_proto_dist_j,
    #                 protoL_rf_info,
    #             )

    #             # get the whole image
    #             original_img_j = search_batch_images[rf_prototype_j[0]]
    #             original_img_j = original_img_j.numpy()
    #             original_img_j = np.transpose(original_img_j, (1, 2, 0))
    #             original_img_size = original_img_j.shape[0]

    #             # crop out the receptive field
    #             rf_img_j = original_img_j[
    #                 rf_prototype_j[1] : rf_prototype_j[2], rf_prototype_j[3] : rf_prototype_j[4], :
    #             ]

    #             # save the prototype receptive field information (pixel indices in the input image)
    #             proto_rf_boxes[j] = {}
    #             proto_rf_boxes[j]['image_index'] = rf_prototype_j[0] + start_index_of_search_batch
    #             proto_rf_boxes[j]['height_start_index'] = rf_prototype_j[1]
    #             proto_rf_boxes[j]['height_end_index'] = rf_prototype_j[2]
    #             proto_rf_boxes[j]['width_start_index'] = rf_prototype_j[3]
    #             proto_rf_boxes[j]['width_end_index'] = rf_prototype_j[4]
    #             proto_rf_boxes[j]['class_indentities'] = search_labels[rf_prototype_j[0]].tolist()

    #             # find the highly activated region of the original image
    #             proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
    #             # the activation function of the distances is log
    #             proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + self.epsilon))
    #             # upsample the matrix with distances (e.g., (14x14)->(224x224))
    #             upsampled_act_img_j = cv2.resize(
    #                 proto_act_img_j,
    #                 dsize=(original_img_size, original_img_size),
    #                 interpolation=cv2.INTER_CUBIC,
    #             )
    #             # find a high activation ROI (default treshold = 95 %)
    #             proto_bound_j = self.find_high_activation_crop(upsampled_act_img_j)
    #             # crop out the ROI with high activation from the image where the distnce for j
    #             # protoype turned out to be the smallest
    #             # the dimensions' order of original_img_j, e.g., (224, 224, 3)
    #             proto_img_j = original_img_j[
    #                 proto_bound_j[0] : proto_bound_j[1], proto_bound_j[2] : proto_bound_j[3], :
    #             ]

    #             # save the ROI (rectangular boundary of highly activated region)
    #             # the activated region can be larger than the receptive field of the patch with the
    #             # smallest distance
    #             proto_bound_boxes[j] = {}
    #             proto_bound_boxes[j]['image_index'] = proto_rf_boxes[j]['image_index']
    #             proto_bound_boxes[j]['height_start_index'] = proto_bound_j[0]
    #             proto_bound_boxes[j]['height_end_index'] = proto_bound_j[1]
    #             proto_bound_boxes[j]['width_start_index'] = proto_bound_j[2]
    #             proto_bound_boxes[j]['width_end_index'] = proto_bound_j[3]
    #             proto_bound_boxes[j]['class_indentities'] = search_labels[
    #                 rf_prototype_j[0]
    #             ].tolist()

    #             # SAVING BLOCK (can be changed later)
    #             if dir_for_saving_prototypes is not None:
    #                 if prototype_self_act_filename_prefix is not None:
    #                     # save the numpy array of the prototype activation map (e.g., (14x14))
    #                     np.save(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_self_act_filename_prefix + str(j) + '.npy',
    #                         ),
    #                         proto_act_img_j,
    #                     )

    #                 if prototype_img_filename_prefix is not None:
    #                     # save the whole image containing the prototype as png
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix + '-original' + str(j) + '.png',
    #                         ),
    #                         original_img_j,
    #                         cmap='gray',
    #                     )

    #                     # overlay (upsampled) activation on original image and save the result
    #                     # normalize the image
    #                     rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
    #                     rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
    #                     heatmap = cv2.applyColorMap(
    #                         np.uint8(255 * rescaled_act_img_j),
    #                         cv2.COLORMAP_JET,
    #                     )
    #                     heatmap = np.float32(heatmap) / 255
    #                     heatmap = heatmap[..., ::-1]
    #                     overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix
    #                             + '-original_with_self_act'
    #                             + str(j)
    #                             + '.png',
    #                         ),
    #                         overlayed_original_img_j,
    #                         vmin=0.0,
    #                         vmax=1.0,
    #                     )

    #                     # if different from the original (whole) image, save the prototype receptive
    #                     # field as png
    #                     if (
    #                         rf_img_j.shape[0] != original_img_size
    #                         or rf_img_j.shape[1] != original_img_size
    #                     ):
    #                         plt.imsave(
    #                             os.path.join(
    #                                 dir_for_saving_prototypes,
    #                                 prototype_img_filename_prefix
    #                                 + '-receptive_field'
    #                                 + str(j)
    #                                 + '.png',
    #                             ),
    #                             rf_img_j,
    #                             vmin=0.0,
    #                             vmax=1.0,
    #                         )
    #                         overlayed_rf_img_j = overlayed_original_img_j[
    #                             rf_prototype_j[1] : rf_prototype_j[2],
    #                             rf_prototype_j[3] : rf_prototype_j[4],
    #                         ]
    #                         plt.imsave(
    #                             os.path.join(
    #                                 dir_for_saving_prototypes,
    #                                 prototype_img_filename_prefix
    #                                 + '-receptive_field_with_self_act'
    #                                 + str(j)
    #                                 + '.png',
    #                             ),
    #                             overlayed_rf_img_j,
    #                             vmin=0.0,
    #                             vmax=1.0,
    #                         )

    #                     # save the highly activated ROI (highly activated region of the whole image)
    #                     plt.imsave(
    #                         os.path.join(
    #                             dir_for_saving_prototypes,
    #                             prototype_img_filename_prefix + str(j) + '.png',
    #                         ),
    #                         proto_img_j,
    #                         vmin=0.0,
    #                         vmax=1.0,
    #                     )


def _warm(**kwargs) -> None:
    for arg in kwargs.values():
        arg.warm()


def _joint(**kwargs) -> None:
    for arg in kwargs.values():
        arg.joint()


def _last(**kwargs) -> None:
    for arg in kwargs.values():
        arg.last()


def _print_status_bar(trainer: pl.Trainer, blocks: Dict, status: str = '') -> None:
    trainer.progress_bar_callback.status_bar.set_description_str(
        '{status}, REQUIRES GRAD: Encoder ({encoder}), Transient layers ({trans}),'
        ' Protorype layer ({prot}), Last layer ({last})'.format(
            status=status,
            encoder=_get_grad_status(blocks['encoder']),
            trans=_get_grad_status(blocks['transient_layers']),
            # protoype layer is inhereted from Tensor and, therefore, has the requires_grad attr
            prot=blocks['prototype_layer'].requires_grad,
            last=_get_grad_status(blocks['last_layer']),
        ),
    )


def _get_grad_status(block: nn.Module) -> bool:
    first_param = next(block.parameters())
    if all(param.requires_grad == first_param.requires_grad for param in block.parameters()):
        return first_param.requires_grad
    else:
        raise Exception(
            f'Not all the parmaters in {block.__class__.__name__} have the same grad status',
        )


def _save_new_checkpoint(path_best_model: str, trainer: pl.Trainer, f1_val: float = 0) -> None:
    os.remove(path_best_model)
    new_checkpoint = trainer.checkpoint_callback.format_checkpoint_name(
        dict(epoch=trainer.current_epoch, step=trainer.global_step, f1_val=f1_val),
    )
    trainer.save_checkpoint(new_checkpoint)
    trainer.checkpoint_callback.best_model_path = new_checkpoint
