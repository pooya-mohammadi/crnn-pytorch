import os
from argparse import ArgumentParser
from os.path import join
from os.path import split

import albumentations
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from deep_utils import split_extension, log_print
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class CRNNDataset(Dataset):

    def __init__(self, root, characters, transform=None, logger=None):
        self.transform = transform
        # index zero is reserved for CTC's delimiter
        self.char2label = {char: i + 1 for i, char in enumerate(characters)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.image_paths, self.labels, self.labels_length = self.get_image_paths(root, characters,
                                                                                 chars2label=self.char2label,
                                                                                 logger=logger)
        self.n_classes = len(self.label2char) + 1  # + 1 is representative of CTC's delimiter!

    @staticmethod
    def text2label(char2label: dict, text: str):
        return [char2label[t] for t in text]

    @staticmethod
    def get_image_paths(root, chars, chars2label, logger=None):
        paths, labels, labels_length = [], [], []
        discards = 0
        for img_name in os.listdir(root):
            img_path = join(root, img_name)
            try:
                if split_extension(img_name)[-1].lower() in ['.jpg', '.png', '.jpeg']:
                    text = CRNNDataset.get_label(img_path)
                    is_valid = CRNNDataset.check_validity(text, chars)
                    if is_valid:
                        label = CRNNDataset.text2label(chars2label, text)
                        labels.append(label)
                        paths.append(img_path)
                        labels_length.append(len(label))
                    else:
                        log_print(logger, f"[Warning] text for sample: {img_path} is invalid. Skipping...")
                        discards += 1
                else:
                    log_print(logger, f"[Warning] sample: {img_path} does not have a valid extension. Skipping...")
                    discards += 1
            except:
                log_print(logger, f"[Warning] sample: {img_path} is not valid. Skipping...")
                discards += 1
        assert len(labels) == len(paths)
        log_print(logger, f"Successfully gathered {len(labels)} samples and discarded {discards} samples!")

        return paths, labels, labels_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.image_paths[index]
        if isinstance(self.transform, albumentations.core.composition.Compose):
            # img = cv2.imread(img_path)[..., ::-1]  # this is used for albumentation
            img = np.array(Image.open(img_path))[..., :3]
            img = self.transform(image=img)['image'][0:1, ...].unsqueeze(0)  # albumentation
        else:
            img = Image.open(img_path)[..., :3]  # This is used for transformers
            img = self.transform(img).unsqueeze(0)  # torch transformers
        label = torch.LongTensor(self.labels[index]).unsqueeze(0)
        label_length = torch.LongTensor([self.labels_length[index]]).unsqueeze(0)

        return img, label, label_length

    @staticmethod
    def get_label(img_path):
        label = split_extension(split(img_path)[-1])[0]
        label = label.split('_')[-1]
        return label

    @staticmethod
    def check_validity(text, chars):
        for c in text:
            if c not in chars:
                return False
        return True

    @staticmethod
    def collate_fn(batch):
        images, labels, labels_lengths = zip(*batch)
        images = torch.cat(images, dim=0)
        labels = [label.squeeze(0) for label in labels]
        # padding with -100, does not matter because they will be ignored by ctc, the labels' length will inform
        # ctc about the valid and padded labels
        labels = nn.utils.rnn.pad_sequence(labels, padding_value=-100).T
        labels_lengths = torch.cat(labels_lengths, dim=0)
        return images, labels, labels_lengths


def get_mean_std(dataset_dir, alphabets, batch_size, img_h, img_w):
    """
    Getting channel wise mean and std
    :return:
    """
    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()]
    )

    dataset = CRNNDataset(root=dataset_dir, transform=transformations, characters=alphabets)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    mean, std = 0, 0
    n_samples = len(dataset)
    for images, labels, labels_lengths in tqdm(data_loader, desc="Getting mean and std"):
        # channel wise
        mean += torch.sum(torch.mean(images, dim=(2, 3)), dim=0)
        std += torch.sum(torch.std(images, dim=(2, 3)), dim=0)
    mean /= n_samples
    std /= n_samples
    return [round(m, 4) for m in mean.numpy().tolist()], [round(s, 4) for s in std.numpy().tolist()]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="path to dataset")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--alphabets", default='ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹', help="alphabets used in dataset")
    parser.add_argument("--img_h", default=32, type=int)
    parser.add_argument("--img_w", default=100, type=int)
    args = parser.parse_args()

    mean, std = get_mean_std(args.dataset_dir, alphabets=args.alphabets, batch_size=args.batch_size,
                             img_h=args.img_h, img_w=args.img_w)
    log_print(None, f"MEAN: {mean}, STD: {std}")
