"""
This code is responsible for getting unique characters to be used as outputs of CTC loss.
"""
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_directory", default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/fa_dataset")
args = parser.parse_args()

characters = set()
for name in os.listdir(args.data_directory):
    characters |= set(os.path.splitext(name)[0].split('_')[0])
print(f'[INFO] characters: {"".join(sorted(characters))}')
