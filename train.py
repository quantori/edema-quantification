from torch.utils.data import DataLoader, TensorDataset
import torch
import pytorch_lightning as pl

from tools.data_classes import EdemaDataModule
from models_edema import EdemaNet
from prototype_model_utils import PNetProgressBar
from pm_settings import EdemaNetSettings


if __name__ == '__main__':
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create the model
    edema_net_st = EdemaNet(settings=EdemaNetSettings())
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    datamaodlule = EdemaDataModule(
        data_dir='C:/temp/edema/edema-quantification/dataset/MIMIC-CXR-Edema-Intermediate',
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
