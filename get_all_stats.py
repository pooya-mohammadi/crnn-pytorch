""" Get all the required stats
"""
from argparse import ArgumentParser
from dataset import get_mean_std
from get_optimum_img_w import get_optimum_img_w
from get_character_sets import get_unique_characters

parser = ArgumentParser()
parser.add_argument("--data_directory", required=True, type=str, help="path to dataset")
parser.add_argument("--batch_size", default=128)
parser.add_argument("--img_h", default=32, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    alphabets = get_unique_characters(args.data_directory)
    max_length, optimal_width = get_optimum_img_w(args.data_directory, alphabets)
    mean, std = get_mean_std(args.data_directory, alphabets, args.batch_size, args.img_h, optimal_width)
    print(
        f"[INFO] Stats: alphabets: {alphabets}, img_w: {optimal_width}, mean: {mean}, std: {std},"
        f" n_classes: {len(alphabets) + 1}, max_length: {max_length}")
