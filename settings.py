import torch
# from torchvision import transforms
import albumentations as A
from dataclasses import dataclass
from albumentations.pytorch import ToTensorV2
from alphabets import ALPHABETS
from argparse import Namespace


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

    def update_basic(self):
        self.n_classes = len(self.alphabets) + 1


@dataclass(init=True, repr=True)
class AugConfig(BasicConfig):
    # train_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    # )
    # val_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Scale()
    #     transforms.Resize((BasicConfig.img_h, BasicConfig.img_w)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=BasicConfig.mean, std=BasicConfig.std), ]
    # )

    train_transform = A.Compose(
        [A.Rotate(limit=10, p=0.2),
         A.RandomScale(scale_limit=0.2),
         A.Resize(height=BasicConfig.img_h, width=BasicConfig.img_w),
         A.Normalize(BasicConfig.mean, BasicConfig.std, max_pixel_value=255.0),
         A.ToGray(always_apply=True, p=1),
         ToTensorV2()
         ])
    val_transform = A.Compose(
        [
         A.Resize(height=BasicConfig.img_h, width=BasicConfig.img_w),
         A.Normalize(BasicConfig.mean, BasicConfig.std, max_pixel_value=255.0),
         A.ToGray(always_apply=True, p=1),
         ToTensorV2()
         ])

    def update_aug(self):
        # self.train_transform = transforms.Compose([
        #     transforms.Grayscale(),
        #     transforms.Resize((self.img_h, self.img_w)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.mean, std=self.std), ]
        # )
        # self.val_transform = transforms.Compose([
        #     transforms.Grayscale(),
        #     transforms.Resize((self.img_h, self.img_w)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.mean, std=self.std), ]
        # )
        self.train_transform = A.Compose(
            [A.Rotate(limit=10, p=0.2),
             A.RandomScale(scale_limit=0.2),
             A.Resize(height=self.img_h, width=self.img_w),
             A.Normalize(self.mean, self.std, max_pixel_value=255.0),
             A.ToGray(always_apply=True, p=1),
             ToTensorV2()
             ])
        self.val_transform = A.Compose(
            [A.Resize(height=self.img_h, width=self.img_w),
             A.Normalize(self.mean, self.std, max_pixel_value=255.0),
             A.ToGray(always_apply=True, p=1),
             ToTensorV2()
             ])


@dataclass(init=True)
class Config(AugConfig):
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
        if isinstance(args, Namespace):
            variables = vars(args)
        elif isinstance(args, dict):
            variables = args
        else:
            raise ValueError()
        for k, v in variables.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k == "visualize":
                print("[INFO] Skipped visualize argument!")
            else:
                raise ValueError(f"value {k} is not defined in Config...")
        self.update()

    def update(self):
        self.char2label = {char: i + 1 for i, char in enumerate(self.alphabets)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.update_basic()
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
