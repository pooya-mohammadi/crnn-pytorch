from argparse import ArgumentParser
import cv2
from PIL import Image
from deep_utils import CRNNInferenceTorch, split_extension, Box
import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--img_path", default="sample_images/image_01.jpg")
    parser.add_argument("--save_img", action="store_true", help="if set saves the output image")
    args = parser.parse_args()
    model = CRNNInferenceTorch(args.model_path)
    img = Image.open(args.img_path)
    tic = time.time()
    prediction = model.infer(args.img_path)
    prediction = "".join(prediction)
    toc = time.time()
    if args.save_img:
        img = cv2.imread(args.img_path)
        img = Box.put_text_pil(img, prediction, org=(20, 20), font="assets/Vazir.ttf", font_size=32)
        cv2.imwrite(split_extension(args.img_path, suffix="_res"), img)
    print("prediction:", "".join(prediction), f"\n elapsed time is {toc - tic}")
