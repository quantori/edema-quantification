import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.data.data_classes import EdemaDataModule
from src.models.prototype_model.models_edema import EdemaNet
from src.models.prototype_model.prototype_model_utils import PNetProgressBar


@hydra.main(version_base=None, config_path='pm_configs', config_name='config')
def main(cfg: DictConfig):
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create the model
    edema_net_st = EdemaNet(settings=cfg.model)
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    datamaodlule = EdemaDataModule(
        data_dir='C:/temp/edema/edema-quantification/data/interim',
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
