import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.data.data_classes import EdemaDataModule
from src.models.prototype_model.model_edema import EdemaPrototypeNet
from src.models.prototype_model.utils import PNetProgressBar
from encoders import ENCODERS
from transient_layers import TransientLayers
from prototype_layers import PrototypeLayer
from last_layers import LastLayers
from loggers import PrototypeLoggerCompNumpy


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='pm_config',
    version_base=None,
)
def main(cfg: DictConfig):
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create blocks and model
    encoder = ENCODERS['squezee_net']()
    transient_layers = TransientLayers(encoder, cfg.model.prototype_shape)
    prototype_layer = PrototypeLayer(
        cfg.model.num_classes,
        cfg.model.num_prototypes,
        cfg.model.prototype_shape,
        cfg.model.prototype_layer_stride,
        cfg.model.epsilon,
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
        batch_size=16,
        resize=(400, 400),
        normalize_tensors=False,
    )
    datamaodlule.setup('fit')
    train_dataloader = datamaodlule.train_dataloader(num_workers=4)
    test_dataloader = datamaodlule.test_dataloader(num_workers=4)

    # create trainer and start training
    trainer = pl.Trainer(
        max_epochs=10,
        logger=True,
        enable_checkpointing=False,
        gpus=1,
        log_every_n_steps=5,
        callbacks=[PNetProgressBar()],
    )
    trainer.fit(edema_net, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
