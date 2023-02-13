from torch.utils.data import DataLoader, TensorDataset
import torch
import pytorch_lightning as pl

from tools.data_classes import EdemaDataModule
from models_edema import SqueezeNet, EdemaNet
from prototype_model_utils import PNetProgressBar


if __name__ == '__main__':
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create the model
    img_size = 400
    sq_net = SqueezeNet()
    edema_net_st = EdemaNet(
        encoder=sq_net, num_classes=9, prototype_shape=(9, 512, 1, 1), img_size=img_size
    )
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    datamaodlule = EdemaDataModule(
        data_dir='C:/temp/edema/edema-quantification/dataset/MIMIC-CXR-Edema-Intermediate',
        batch_size=16,
        resize=(img_size, img_size),
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
