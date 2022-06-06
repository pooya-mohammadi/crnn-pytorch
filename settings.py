import torch
from torchvision import transforms

from alphabets import ALPHABETS


class BasicConfig:
    IMG_H = 32  # the height of the input image to network
    IMG_W = 100  # the width of the input image to network

    FILE_NAME = "best"

    # Modify
    N_CLASSES = 35
    MEAN = [0.4845]
    STD = [0.1884]
    ALPHABET_NAME = "FA_LPR"
    TRAIN_ROOT = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train'
    VAL_ROOT = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val'
    OUTPUT_DIR = "output"


class Config(BasicConfig):
    N_HIDDEN = 256  # size of the lstm hidden state
    N_CHANNELS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 0.0005
    LR_PATIENCE = 10
    LR_REDUCE_FACTOR = 0.1
    BATCH_SIZE = 128
    EPOCHS = 200
    N_WORKERS = 8

    TRANSFORMATION = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((BasicConfig.IMG_H, BasicConfig.IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=BasicConfig.MEAN, std=BasicConfig.STD), ]
    )

    ALPHABETS = ALPHABETS[BasicConfig.ALPHABET_NAME]
    CHAR2LABEL = dict()
    LABEL2CHAR = dict()

    # Early stopping
    EARLY_STOPPING_PATIENCE = 30


Config.CHAR2LABEL = {char: i + 1 for i, char in enumerate(Config.ALPHABETS)}
Config.LABEL2CHAR = {label: char for char, label in Config.CHAR2LABEL.items()}
