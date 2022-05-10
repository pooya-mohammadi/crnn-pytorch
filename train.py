import torch
import pytorch_lightning as pl
from dataset import CRNNDataset
from crnn import CRNN
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

torch.backends.cudnn.benchmark = True


class LitCRNN(pl.LightningModule):
    def __init__(self):
        super(LitCRNN, self).__init__()
        self.model = CRNN(img_h=Config.IMG_H,
                          n_channels=Config.N_CHANNELS,
                          n_classes=Config.N_CLASSES,
                          n_hidden=Config.N_HIDDEN)
        self.model.apply(self.model.weights_init)
        self.criterion = CTCLoss(reduction='sum')

    def forward(self, x):
        logit = self.model(x)
        logit = torch.transpose(logit, 1, 0)
        return logit

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, labels, labels_lengths = batch

        labels_lengths = labels_lengths.squeeze(1)
        batch_size = images.size(0)
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        loss = self.criterion(logits, labels, input_lengths, labels_lengths) / batch_size
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, labels_lengths = batch

        labels_lengths = labels_lengths.squeeze(1)
        batch_size = images.size(0)
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        loss = self.criterion(logits, labels, input_lengths, labels_lengths) / batch_size
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.LR)
        return optimizer

    @staticmethod
    def get_loaders():
        train_dataset = CRNNDataset(root=Config.TRAIN_ROOT, characters=Config.ALPHABETS,
                                    transform=Config.TRANSFORMATION)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Config.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=Config.WORKERS,
                                                   collate_fn=train_dataset.collate_fn
                                                   )

        val_dataset = CRNNDataset(root=Config.VAL_ROOT, characters=Config.ALPHABETS, transform=Config.TRANSFORMATION)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=True,
                                                 batch_size=Config.BATCH_SIZE,
                                                 num_workers=Config.WORKERS,
                                                 collate_fn=val_dataset.collate_fn)

        return train_loader, val_loader


def main():
    early_stopping = EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE)
    model_checkpoint = ModelCheckpoint(dirpath=Config.MODEL_PATH, filename=Config.FILE_NAME, monitor="val_loss",
                                       verbose=True)
    trainer = pl.Trainer(gpus=1 if Config.DEVICE == "cuda" else 0,
                         max_epochs=Config.EPOCHS,
                         min_epochs=Config.EPOCHS // 10,
                         callbacks=[early_stopping, model_checkpoint])
    lit_crnn = LitCRNN()
    train_loader, val_loader = lit_crnn.get_loaders()
    trainer.fit(model=lit_crnn, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
