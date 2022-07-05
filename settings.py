import torch
from torchvision import transforms
from dataclasses import dataclass
from alphabets import ALPHABETS


@dataclass(init=True)
class BasicConfig:
    IMG_H = 32  # the height of the input image to network
    IMG_W = 100  # the width of the input image to network

    FILE_NAME = "best"

    # Modify
    N_CLASSES = 35
    mean = [0.4845]
    std = [0.1884]
    ALPHABET_NAME = "FA_LPR"
    train_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train'
    val_directory = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val'
    output_dir = "output"


@dataclass(init=True)
class Config(BasicConfig):
    N_HIDDEN = 256  # size of the lstm hidden state
    N_CHANNELS = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 0.0005
    LR_PATIENCE = 10
    LR_REDUCE_FACTOR = 0.1
    BATCH_SIZE = 128
    epochs = 200
    N_WORKERS = 8

    TRANSFORMATION = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((BasicConfig.IMG_H, BasicConfig.IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    )

    alphabets = ALPHABETS[BasicConfig.ALPHABET_NAME]
    CHAR2LABEL = dict()
    LABEL2CHAR = dict()

    # Early stopping
    EARLY_STOPPING_PATIENCE = 30

    def update_config_param(self, args):
        variables = vars(args)
        for k, v in variables.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"value {k} is not defined in Config...")
        self.update()

    def update(self):
        self.CHAR2LABEL = {char: i + 1 for i, char in enumerate(Config.alphabets)}
        self.LABEL2CHAR = {label: char for char, label in self.CHAR2LABEL.items()}

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
