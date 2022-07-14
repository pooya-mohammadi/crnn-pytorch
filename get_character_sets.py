"""
This code is responsible for getting unique characters to be used as outputs of CTC loss.
"""
import os
from argparse import ArgumentParser
from dataset import CRNNDataset

parser = ArgumentParser()
parser.add_argument("--data_directory", default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/fa_dataset")
args = parser.parse_args()

characters = set()
for name in os.listdir(args.data_directory):
    characters |= set(CRNNDataset.get_label(name))
print(f'[INFO] characters: {"".join(sorted(characters))}')
