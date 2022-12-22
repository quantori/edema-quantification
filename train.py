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
    test_dataset = TensorDataset(
        torch.rand(128, 10, 350, 350), torch.randint(0, 2, (128, 7), dtype=torch.float32)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # create the trainer and start training
    trainer = pl.Trainer(max_epochs=9, logger=False, enable_checkpointing=False, gpus=1)
    trainer.fit(edema_net, test_dataloader)
