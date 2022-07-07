import torch
from torchvision import transforms
from dataclasses import dataclass
from alphabets import ALPHABETS


@dataclass(init=True)
class BasicConfig:
    img_h = 32  # the height of the input image to network
    img_w = 100  # the width of the input image to network

    file_name = "best"

    # Modify
    n_classes = 35
    mean = [0.4845]
    std = [0.1884]
    alphabet_name = "FA_LPR"
    train_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train'
    val_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val'
    output_dir = "output"


@dataclass(init=True, repr=True)
class AugConfig:
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    )
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    )

    def update_aug(self):
        self.train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std), ]
        )
        self.val_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std), ]
        )


@dataclass(init=True)
class Config(BasicConfig, AugConfig):
    n_hidden = 256  # size of the lstm hidden state
    lstm_input = 64  # size of the lstm_input feature size
    n_channels = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.0005
    lr_patience = 10
    lr_reduce_factor = 0.1
    batch_size = 128
    epochs = 200
    n_workers = 8

    alphabets = ALPHABETS[BasicConfig.alphabet_name]
    char2label = dict()
    label2char = dict()

    # Early stopping
    early_stopping_patience = 30

    def update_config_param(self, args):
        variables = vars(args)
        for k, v in variables.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"value {k} is not defined in Config...")
        self.update()

    def update(self):
        self.char2label = {char: i + 1 for i, char in enumerate(self.alphabets)}
        self.label2char = {label: char for char, label in self.char2label.items()}

        self.update_aug()

    def __repr__(self):
        variables = vars(self)
        return f"{self.__class__.__name__} -> " + ", ".join(f"{k}: {v}" for k, v in variables.items())

    def vars(self) -> dict:
        out = dict()
        for key in dir(self):
            val = getattr(self, key)
            if (key.startswith("__") and key.endswith("__")) or type(val).__name__ == "method":
                continue
            else:
                out[key] = val
        return out
