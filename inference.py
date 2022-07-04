from argparse import ArgumentParser
from pathlib import Path
from typing import Union, List
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import LitCRNN
from deep_utils import CTCDecoder, split_extension, Box, TorchVisionUtils
import time


class CRNNInference:
    def __init__(self, model_path, decode_method='greedy', device='cpu'):
        self.device = device
        self.decode_method = decode_method
        self.model = LitCRNN.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.label2char = state_dict['label2char']
        self.transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((state_dict['img_height'], state_dict['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=state_dict['mean'], std=state_dict['std'])]
        )
        del state_dict

    def infer(self, img: Union[str, Path, np.ndarray]):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = Image.open(img)
        image = self.transformer(img)
        image = image.view(1, *image.size())
        with torch.no_grad():
            preds = self.model(image).cpu().squeeze(0).numpy()
        sim_pred = CTCDecoder.ctc_decode(preds, decoder_name=self.decode_method, label2char=self.label2char)
        return sim_pred

    def infer_group(self, images: Union[List[np.ndarray]]):
        images = TorchVisionUtils.transform_concatenate_images(images, self.transformer, device=self.device)
        with torch.no_grad():
            preds = self.model(images).squeeze(0).numpy()
        sim_preds = CTCDecoder.ctc_decode_batch(preds, decoder_name=self.decode_method, label2char=self.label2char)
        return sim_preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--img_path", default="sample_images/image_01.jpg")
    args = parser.parse_args()
    model = CRNNInference(args.model_path)
    img = Image.open(args.img_path)
    tic = time.time()
    prediction = model.infer(args.img_path)
    prediction = "".join(prediction)
    toc = time.time()
    img = cv2.imread(args.img_path)
    img = Box.put_text_pil(img, prediction, org=(20, 20), font="assets/Vazir.ttf", font_size=32)
    cv2.imwrite(split_extension(args.img_path, suffix="_res"), img)
    print("prediction:", "".join(prediction), f"\n elapsed time is {toc - tic}")
