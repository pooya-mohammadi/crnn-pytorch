from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from alphabets import ALPHABETS
from train import LitCRNN
from deep_utils import CTCDecoder, Box, show_destroy_cv2


class CRNNPred:
    def __init__(self, model_path, characters, img_height=32, img_width=100,
                 decode_method='greedy'):
        self.char2label = {char: i + 1 for i, char in enumerate(characters)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.decode_method = decode_method
        self.model = LitCRNN.load_from_checkpoint(model_path)
        self.model.eval()

        self.transformer = T.Compose([
            T.Grayscale(),
            T.RandomHorizontalFlip(p=1),
            T.Resize((img_height, img_width)),
            T.ToTensor()]
        )

    def detect(self, img: Union[str, Path, np.ndarray]):
        if type(img) is np.ndarray:
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
    parser.add_argument("--model_path", default="logs/best_model.ckpt")
    parser.add_argument("--alphabet_name", default="FA_LPR", help="alphabet name from alphabets.py module")
    parser.add_argument("--img_path", default="sample_images/۱۳ج۷۷۲۴۴_9779.jpg")
    args = parser.parse_args()
    model = CRNNPred(args.model_path, characters=ALPHABETS[args.alphabet_name])
    img = Image.open(args.img_path)
    prediction = model.detect(args.img_path)
    prediction = "".join(prediction)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("assets/Vazir.ttf", 32)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((20, 20), prediction, (0, 255, 0), font=font)

    show_destroy_cv2(np.array(img)[..., ::-1])
    print("".join(prediction))
