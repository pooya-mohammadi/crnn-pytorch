from os.path import join
import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CRNNDataset
from crnn import CRNN
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

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

    def get_loss(self, batch):
        images, labels, labels_lengths = batch
        labels_lengths = labels_lengths.squeeze(1)
        batch_size = images.size(0)
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        loss = self.criterion(logits, labels, input_lengths, labels_lengths)
        # if train:
        #     acc = 0
        # else:
        #     preds = torch.transpose(logits, 1, 0).detach().cpu().numpy()
        #     preds = CTCDecoder.ctc_decode(preds, decoder_name=self.decode_method, label2char=None)
        #     acc = sum("-".join([str(p) for p in pred]) == "-".join([str(l) for l in labels]) for pred, label in
        #               zip(preds, labels.detach().cpu().numpy()))
        return loss, batch_size

    @staticmethod
    def calculate_metrics(outputs):
        r_loss, size = 0, 0
        for row in outputs:
            r_loss += row["loss"]
            size += row["bs"]
        loss = r_loss / size
        return loss

    def test_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def training_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def training_epoch_end(self, outputs) -> None:
        loss = self.calculate_metrics(outputs)
        self.log("train_loss", loss.item())

    def validation_epoch_end(self, outputs) -> None:
        loss = self.calculate_metrics(outputs)
        self.log("val_loss", loss.item())

    def test_epoch_end(self, outputs) -> None:
        loss = self.calculate_metrics(outputs)
        self.log("test_loss", loss.item())

    def validation_step(self, batch, batch_idx):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.LR)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.LR_REDUCE_FACTOR,
                                      patience=Config.LR_PATIENCE, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def get_loaders():
        train_dataset = CRNNDataset(root=Config.TRAIN_ROOT, characters=Config.ALPHABETS,
                                    transform=Config.TRANSFORMATION)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Config.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=Config.N_WORKERS,
                                                   collate_fn=train_dataset.collate_fn
                                                   )

        val_dataset = CRNNDataset(root=Config.VAL_ROOT, characters=Config.ALPHABETS, transform=Config.TRANSFORMATION)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=True,
                                                 batch_size=Config.BATCH_SIZE,
                                                 num_workers=Config.N_WORKERS,
                                                 collate_fn=val_dataset.collate_fn)

        return train_loader, val_loader


def main():
    output_dir = mkdir_incremental(Config.OUTPUT_DIR)
    early_stopping = EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir, filename=Config.FILE_NAME, monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if Config.DEVICE == "cuda" else 0,
                         max_epochs=Config.EPOCHS,
                         min_epochs=Config.EPOCHS // 10,
                         callbacks=[early_stopping, model_checkpoint, learning_rate_monitor],
                         default_root_dir=output_dir)
    lit_crnn = LitCRNN()
    train_loader, val_loader = lit_crnn.get_loaders()
    trainer.fit(model=lit_crnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(lit_crnn, ckpt_path="best", dataloaders=val_loader)
    trainer.test(lit_crnn, ckpt_path="best", dataloaders=train_loader)

    # Adding artifacts to weights
    weight_path = join(output_dir, f"{Config.FILE_NAME}.ckpt")
    best_weight = torch.load(weight_path)
    best_weight['img_height'] = Config.IMG_H
    best_weight['img_width'] = Config.IMG_W
    best_weight['n_channels'] = Config.N_CHANNELS
    best_weight['n_hidden'] = Config.N_HIDDEN
    best_weight['mean'] = Config.MEAN
    best_weight['std'] = Config.STD
    best_weight['label2char'] = Config.LABEL2CHAR
    best_weight['n_classes'] = Config.N_CLASSES
    torch.save(best_weight, weight_path)


if __name__ == '__main__':
    main()
