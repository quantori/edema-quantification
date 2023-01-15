from torch.utils.data import DataLoader, TensorDataset
import torch
import pytorch_lightning as pl

from tools.data_classes import EdemaDataModule, EdemaDataset
from models_edema import SqueezeNet, EdemaNet


if __name__ == '__main__':
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create a model
    sq_net = SqueezeNet()
    edema_net_st = EdemaNet(sq_net, 9, prototype_shape=(18, 512, 1, 1))
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    datamaodlule = EdemaDataModule(
        data_dir='C:/Users/makov/Desktop/data_edema', resize=(300, 300), normalize_tensors=False
    )
    datamaodlule.setup('fit')
    train_dataloader = datamaodlule.train_dataloader(num_workers=4)
    test_dataloader = datamaodlule.test_dataloader(num_workers=4)

    # create the trainer and start training
    trainer = pl.Trainer(max_epochs=9, logger=True, enable_checkpointing=False, gpus=1)
    trainer.fit(edema_net, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
