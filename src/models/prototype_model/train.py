import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from src.data.data_classes import EdemaDataModule
from model_edema import EdemaPrototypeNet
from utils import PNetProgressBar
from encoders import ENCODERS
from transient_layers import TransientLayers
from prototype_layers import PrototypeLayer
from last_layers import LastLayers
from loggers import PrototypeLoggerCompNumpy


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='pm_config',
    version_base=None,
)
def main(cfg: DictConfig):
    # Clean the gpu cache
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)

    # Create blocks and model
    encoder = ENCODERS['squezee_net']()
    transient_layers = TransientLayers(encoder, cfg.model.prototype_shape)
    prototype_layer = PrototypeLayer(
        cfg.model.num_classes,
        cfg.model.num_prototypes,
        cfg.model.prototype_shape,
        prototype_layer_stride=cfg.model.prototype_layer_stride,
        epsilon=cfg.model.epsilon,
    )
    last_layers = LastLayers(cfg.model.num_prototypes, cfg.model.num_classes, bias=False)
    prototype_logger = PrototypeLoggerCompNumpy(logger_config=cfg.logger)
    edema_net_st = EdemaPrototypeNet(
        encoder, transient_layers, prototype_layer, last_layers, cfg.model, prototype_logger
    )
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    datamaodlule = EdemaDataModule(
        data_dir='data/interim',
        batch_size=cfg.model.batch_size,
        resize=(cfg.model.img_size, cfg.model.img_size),
        normalize_tensors=False,
    )
    datamaodlule.setup('fit')
    train_dataloader = datamaodlule.train_dataloader(num_workers=cfg.model.num_cpu)
    test_dataloader = datamaodlule.test_dataloader(num_workers=cfg.model.num_cpu)

    # create model checkpoint and trainer and start training
    checkpoint = ModelCheckpoint(
        filename='{epoch}-{step}-{f1_val:.3f}',
        monitor='f1_val',
        mode='max',
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        max_epochs=83,
        logger=True,
        enable_checkpointing=True,
        gpus=1,
        log_every_n_steps=5,
        callbacks=[PNetProgressBar(), checkpoint],
        deterministic=False,
    )
    trainer.fit(edema_net, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
