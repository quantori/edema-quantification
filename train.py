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
    train_dataset = TensorDataset(
        torch.rand(128, 10, 300, 300), torch.randint(0, 2, (128, 7), dtype=torch.float32)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4)

    val_dataset = TensorDataset(
        torch.rand(128, 10, 300, 300), torch.randint(0, 2, (128, 7), dtype=torch.float32)
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # create the trainer and start training
    trainer = pl.Trainer(max_epochs=9, logger=, enable_checkpointing=False, gpus=1)
    trainer.fit(edema_net, train_dataloader, val_dataloaders=val_dataloader)
