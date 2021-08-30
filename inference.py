import time
import torch
from PIL import Image
from recognition.crnn_pytorch import utils, params, dataset, models


class CRNN:
    def __init__(self, model_path, device):
        self.model = models.CRNN(params.imgH,
                                 params.nc,
                                 len(params.alphabet) + 1,
                                 params.nh).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.converter = utils.strLabelConverter(params.alphabet)
        self.transformer = dataset.ResizeNormalize((params.imgW, params.imgH))
        self.device = device

    def detect(self, img):
        tic = time.time()
        img = Image.fromarray(img).convert('L')
        image = self.transformer(img).to(self.device)
        image = image.view(1, *image.size())
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.LongTensor([preds.size(0)])
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        toc = time.time()
        print(toc - tic)
        return sim_pred
