from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from train import LitCRNN
from deep_utils import CTCDecoder, show_destroy_cv2, split_extension, Box
import time


class CRNNPred:
    def __init__(self, model_path, decode_method='greedy'):
        self.decode_method = decode_method
        self.model = LitCRNN.load_from_checkpoint(model_path)
        self.model.eval()
        state_dict = torch.load(model_path)
        self.label2char = state_dict['label2char']
        self.transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((state_dict['img_height'], state_dict['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=state_dict['mean'], std=state_dict['std'])]
        )
        del state_dict

    def detect(self, img: Union[str, Path, np.ndarray]):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = Image.open(img)
        image = self.transformer(img)
        image = image.view(1, *image.size())
        with torch.no_grad():
            preds = self.model(image).squeeze(0).numpy()
        sim_pred = CTCDecoder.ctc_decode(preds, decoder_name=self.decode_method, label2char=self.label2char)
        return sim_pred


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--img_path", default="sample_images/image_01.jpg")
    args = parser.parse_args()
    model = CRNNPred(args.model_path)
    img = Image.open(args.img_path)
    tic = time.time()
    prediction = model.detect(args.img_path)
    prediction = "".join(prediction)
    toc = time.time()
    img = cv2.imread(args.img_path)
    img = Box.put_text_pil(img, prediction, org=(20, 20), font="assets/Vazir.ttf", font_size=32)
    cv2.imwrite(split_extension(args.img_path, suffix="_res"), img)
    print("prediction:", "".join(prediction), f"\n elapsed time is {toc - tic}")
