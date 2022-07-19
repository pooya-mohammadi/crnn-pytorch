"""
This code is responsible for getting unique characters to be used as outputs of CTC loss.
"""
import os
from argparse import ArgumentParser
from dataset import CRNNDataset

parser = ArgumentParser()
parser.add_argument("--data_directory", required=True, type=str, help="path to dataset")


def get_unique_characters(data_directory):
    unique_characters = set()
    for name in os.listdir(data_directory):
        unique_characters |= set(CRNNDataset.get_label(name))
    unique_characters = "".join(sorted(unique_characters))
    return unique_characters


if __name__ == '__main__':
    args = parser.parse_args()
    characters = get_unique_characters(args.data_directory)
    print(f'[INFO] characters: {characters}')
