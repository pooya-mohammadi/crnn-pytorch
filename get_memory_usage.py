from deep_utils import CRNNModelTorch
from settings import Config
from deep_utils import MemoryUtilsTorch

if __name__ == '__main__':
    device = "cuda"
    model = CRNNModelTorch(img_h=Config.img_h, n_channels=Config.n_channels, n_classes=Config.n_classes,
                           n_hidden=Config.n_hidden).to(device).eval()
    MemoryUtilsTorch.memory_test(model, device=device, input_size=(1, Config.n_channels, Config.img_h, Config.img_w),
                                 get_extra_info=False)
