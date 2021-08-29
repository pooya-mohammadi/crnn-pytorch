import torch
import persian_alphabets as alphabets
from deep_utils import AugmentTorch
# about data and net
TRAIN_ROOT, VAL_ROOT = '/home/ai/projects/Irancel/recognition/train', '/home/ai/projects/Irancel/recognition/val'
alphabet = alphabets.alphabet
keep_ratio = False  # whether to keep ratio for image resize
manualSeed = 1234  # reproduce experiemnt
random_sample = True  # whether to sample the hand_labeled_dataset with random sampler
imgH = 32  # the height of the input image to network
imgW = 100  # the width of the input image to network
nh = 256  # size of the lstm hidden state
nc = 1
pretrained = ''  # path to pretrained model (to continue training)
expr_dir = 'expr'  # where to store samples and models
dealwith_lossnan = False  # whether to replace all nan/inf in gradients to zero

# hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # enables cuda
multi_gpu = False  # whether to use multi gpu
ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0  # number of data loading workers

# training process
displayInterval = 100  # interval to be print the train loss
valInterval = 1000  # interval to val the model loss and accuracy
saveInterval = 1000  # interval to save model
n_val_disp = 10  # number of samples to display when val the model

# finetune
nepoch = 2000  # number of epochs to train for
batchSize = 64  # input batch size
lr = 0.0001  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
adam = True  # whether to use adam (default is rmsprop)
adadelta = False  # whether to use adadelta (default is rmsprop)
if nc == 1:
    mean, std = [0.5], [0.5]
else:
    mean, std = [0.5] * nc, [0, 5] * nc

transformations = AugmentTorch.get_augments(
    AugmentTorch.resize((imgH, imgW)),
    AugmentTorch.normalize(mean=mean, std=std),
    AugmentTorch.random_rotation((-20, 20)),
    AugmentTorch.gaussian_blur(kernel_size=3),

)