from crnn import CRNN
from settings import Config
from deep_utils import MemoryUtilsTorch

if __name__ == '__main__':
    device = "cuda"
    model = CRNN(img_h=Config.IMG_H, n_channels=Config.N_CHANNELS, n_classes=Config.N_CLASSES,
                 n_hidden=Config.N_HIDDEN).to(device).eval()
    MemoryUtilsTorch.memory_test(model, device=device, input_size=(1, Config.N_CHANNELS, Config.IMG_H, Config.IMG_W),
                                 get_extra_info=False)
