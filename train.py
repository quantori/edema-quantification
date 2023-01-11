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
    edema_net_st = EdemaNet(sq_net, 9, prototype_shape=(45, 512, 1, 1))
    edema_net = edema_net_st.cuda()

    # pull the dataset and dataloader
    from tools.data_classes import EdemaDataModule, EdemaDataset

    import pandas as pd
    import os

    # metadata_df = pd.read_excel(os.path.join('C:/Users/makov/Desktop/data_edema', 'metadata.xlsx'))
    # edema_dataset = EdemaDataset(metadata_df)
    # images, labels = edema_dataset.__getitem__(10)
    # print(images.shape)
    # print(labels)
    datamaodlule = EdemaDataModule(data_dir='C:/Users/makov/Desktop/data_edema')
    datamaodlule.setup('fit')
    dataloader = datamaodlule.train_dataloader()
    # for i in range(10):
    #     train_features, train_labels = next(iter(dataloader))
    # print(train_features.shape)
    # print(train_labels[20])
    # images = train_features[20]
    # import matplotlib.pyplot as plt

    # images = images[3:]
    # plt.imshow(images[5])
    # plt.show()

    # create the trainer and start training
    trainer = pl.Trainer(max_epochs=9, logger=True, enable_checkpointing=False, gpus=1)
    trainer.fit(edema_net, dataloader, val_dataloaders=dataloader)
