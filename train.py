from argparse import ArgumentParser
from pathlib import Path
import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental, CRNNModelTorch, get_logger, TorchUtils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CRNNDataset
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True


class LitCRNN(pl.LightningModule):
    def __init__(self, img_h, n_channels, n_classes, n_hidden, lstm_input, lr, lr_reduce_factor, lr_patience):
        super(LitCRNN, self).__init__()
        self.save_hyperparameters()
        self.model = CRNNModelTorch(img_h=self.hparams.img_h,
                                    n_channels=self.hparams.n_channels,
                                    n_classes=self.hparams.n_classes,
                                    n_hidden=self.hparams.n_hidden,
                                    lstm_input=self.hparams.lstm_input)
        self.model.apply(self.model.weights_init)
        self.criterion = CTCLoss(reduction='mean')

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_reduce_factor,
                                      patience=self.hparams.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def get_loaders(config):
        train_dataset = CRNNDataset(root=config.train_directory, characters=config.alphabets,
                                    transform=config.train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.n_workers,
                                                   collate_fn=train_dataset.collate_fn
                                                   )

        val_dataset = CRNNDataset(root=config.val_directory, characters=config.alphabets,
                                  transform=config.val_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=True,
                                                 batch_size=config.batch_size,
                                                 num_workers=config.n_workers,
                                                 collate_fn=val_dataset.collate_fn)

        return train_loader, val_loader


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_directory", type=Path,
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train",
                        help="path to the dataset, default: ./dataset")
    parser.add_argument("--val_directory", type=Path,
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val",
                        help="path to the dataset, default: ./dataset")
    parser.add_argument("--output_dir", type=Path, default="./output",
                        help="path to the output directory, default: ./output")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--device", default="cuda", help="what should be the device for training, default is cuda")
    parser.add_argument("--mean", nargs="+", type=float, default=[0.4845], help="dataset channel-wise mean")
    parser.add_argument("--std", nargs="+", type=float, default=[0.1884], help="dataset channel-wise std")
    parser.add_argument("--img_w", type=int, default=100, help="dataset img width size")
    parser.add_argument("--n_workers", type=int, default=8, help="number of workers used for dataset collection")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size number")
    parser.add_argument("--alphabets", default='ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹',
                        help="alphabets used in the process")
    args = parser.parse_args()
    config = Config()
    config.update_config_param(args)

    output_dir = mkdir_incremental(config.output_dir)
    logger = get_logger("pytorch-lightning-image-classification", log_path=output_dir / "log.log")
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir, filename=config.file_name, monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if config.device == "cuda" else 0,
                         max_epochs=config.epochs,
                         min_epochs=config.epochs // 10,
                         callbacks=[early_stopping, model_checkpoint, learning_rate_monitor],
                         default_root_dir=output_dir)
    lit_crnn = LitCRNN(config.img_h, config.n_channels, config.n_classes, config.n_hidden, config.lstm_input, config.lr,
                       config.lr_reduce_factor, config.lr_patience)
    train_loader, val_loader = lit_crnn.get_loaders(config)
    trainer.fit(model=lit_crnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(lit_crnn, ckpt_path="best", dataloaders=val_loader)
    trainer.test(lit_crnn, ckpt_path="best", dataloaders=train_loader)

    # Adding artifacts to weights
    weight_path = output_dir / f"{config.file_name}.ckpt"
    TorchUtils.save_config_to_weight(weight_path, config, logger)


if __name__ == '__main__':
    main()
