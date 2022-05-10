import torch
from torchvision import transforms

from alphabets import ALPHABETS


class BasicConfig:
    IMG_H = 32  # the height of the input image to network
    IMG_W = 100  # the width of the input image to network

    ALPHABET_NAME = "FA_LPR"

    # Modify
    N_CLASSES = 35
    MEAN = [0.4845]
    STD = [0.1884]
    TRAIN_ROOT = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train'
    VAL_ROOT = '/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val'
    MODEL_PATH = "logs"
    FILE_NAME = "best_model"


class Config(BasicConfig):
    N_HIDDEN = 256  # size of the lstm hidden state
    N_CHANNELS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 0.0005
    SEAD = 1234
    BATCH_SIZE = 128
    EPOCHS = 100
    WORKERS = 8

    TRANSFORMATION = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(p=1),  # Makes no difference :D
        transforms.Resize((BasicConfig.IMG_H, BasicConfig.IMG_W)),
        transforms.ToTensor()]
    )

    ALPHABETS = ALPHABETS[BasicConfig.ALPHABET_NAME]

    # Early stopping
    EARLY_STOPPING_PATIENCE = 10

# keep_ratio = False  # whether to keep ratio for image resize
# random_sample = True  # whether to sample the hand_labeled_dataset with random sampler
#
# pretrained = ''  # path to pretrained model (to continue training)
# expr_dir = 'expr'  # where to store samples and models
# # dealwith_lossnan = False  # whether to replace all nan/inf in gradients to zero
#
# # hardware
# multi_gpu = False  # whether to use multi gpu
# ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
# workers = 0  # number of data loading workers
#
# # training process
# displayInterval = 100  # interval to be print the train loss
# valInterval = 1000  # interval to val the model loss and accuracy
# saveInterval = 1000  # interval to save model
# n_val_disp = 10  # number of samples to display when val the model
#
# # finetune
# nepoch = 500  # number of epochs to train for
# batchSize = 64  # input batch size
# # lr = 0.0001  # learning rate for Critic, not used by adadealta
# beta1 = 0.5  # beta1 for adam. default=0.5
# adam = True  # whether to use adam (default is rmsprop)
# adadelta = False  # whether to use adadelta (default is rmsprop)
