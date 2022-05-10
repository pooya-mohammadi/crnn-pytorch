import os
from argparse import ArgumentParser
from os.path import join
from os.path import split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from deep_utils import split_extension, log_print
from tqdm import tqdm
from alphabets import ALPHABETS
from settings import Config


class CRNNDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((Config.IMG_H, Config.IMG_W)),
        transforms.ToTensor()]
    )

    def __init__(self, root, characters, transform=None, logger=None):
        self.transform = self.DEFAULT_TRANSFORM if transform is None else transform
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
        img = Image.open(img_path)
        img = self.transform(img).unsqueeze(0)

        label = torch.LongTensor(self.labels[index]).unsqueeze(0)
        label_length = torch.LongTensor([self.labels_length[index]]).unsqueeze(0)

        return img, label, label_length

    @staticmethod
    def get_label(img_path):
        label = split_extension(split(img_path)[-1])[0]
        label = label.split('_')[0]
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
        labels = torch.cat(labels, dim=0)
        labels_lengths = torch.cat(labels_lengths, dim=0)
        return images, labels, labels_lengths


def get_mean_std(data_loader: DataLoader):
    """
    Getting channel wise mean and std
    :param data_loader:
    :return:
    """
    mean, std = 0, 0
    n_samples = len(data_loader.dataset)
    for images, labels, labels_lengths in tqdm(data_loader, desc="Getting mean and std"):
        # channel wise
        mean += torch.sum(torch.mean(images, dim=(2, 3)), dim=0)
        std += torch.sum(torch.std(images, dim=(2, 3)), dim=0)
    mean /= n_samples
    std /= n_samples
    return [round(m, 4) for m in mean.numpy().tolist()], [round(s, 4) for s in std.numpy().tolist()]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_directory",
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--alphabet_name", default="FA_LPR", help="alphabet name from alphabets.py module")
    args = parser.parse_args()

    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(p=1),  # Makes no difference :D
        transforms.Resize((Config.IMG_H, Config.IMG_W)),
        transforms.ToTensor()]
    )

    dataset = CRNNDataset(root=args.train_directory, transform=transformations,
                          characters=ALPHABETS[args.alphabet_name])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    mean, std = get_mean_std(data_loader)
    log_print(None, f"MEAN: {mean}, STD: {std}")
    log_print(None, f"N_CLASSES: {dataset.n_classes} ---> {''.join(dataset.char2label.keys())}")
