# TODO delete some of the imports below if necessery, when the real datasets are imported
from torch.utils.data import DataLoader, TensorDataset
import torch
import pytorch_lightning as pl

# TODO: import the real dataset
# from dataset import dataset, pldataset
from models_edema import SqueezeNet, EdemaNet

if __name__ == '__main__':
    # clean the gpu cache
    torch.cuda.empty_cache()

    # create a model
    sq_net = SqueezeNet()
    edema_net_st = EdemaNet(sq_net, 7, prototype_shape=(35, 512, 1, 1))
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    from tools.data_classes import EdemaDataModule

    datamaodlule = EdemaDataModule(data_dir='C:/Users/makov/Desktop/data_edema')
    datamaodlule.setup('fit')
    dataloader = datamaodlule.train_dataloader()
    print(dataloader)

    # create the trainer and start training
    # trainer = pl.Trainer(max_epochs=9, logger=True, enable_checkpointing=False, gpus=1)
    # trainer.fit(edema_net, train_dataloader, val_dataloaders=val_dataloader)
