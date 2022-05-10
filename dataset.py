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
from params import IMG_H, IMG_W


class CRNNDataset(Dataset):
    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_H, IMG_W)),
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
        label_length = torch.LongTensor(self.labels_length[index]).unsqueeze(0)

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
    return mean, std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_directory",
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--alphabet_name", default="FA_LPR", help="alphabet name from alphabets.py module")
    args = parser.parse_args()

    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(p=0.5),  # Makes no difference :D
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor()]
    )

    dataset = CRNNDataset(root=args.train_directory, transform=transformations,
                          characters=ALPHABETS[args.alphabet_name])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    mean, std = get_mean_std(data_loader)
    log_print(None, f"MEAN: {mean}, STD: {std}")

#
# def get_train_transform(random_resize_crop=dict(size=None, scale=(0.08, 1.0),
#                                                 ratio=(3. / 4., 4. / 3.),
#                                                 interpolation=InterpolationMode.BILINEAR),
#
#                         ):
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
#         transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([transforms.RandomRotation(rotation_range)], p=0.5),
#         transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5 if gussian_blur else 0),
#         transforms.Pad(2 if crop_size == 48 else 0),
#         transforms.TenCrop(crop_size),
#         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#         transforms.Lambda(
#             lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
#         transforms.Lambda(
#             lambda tensors: torch.stack(
#                 [transforms.RandomErasing(p=0 if cutmix else 0.5)(t) for t in tensors])),
#     ])
#
#
# get_train_transform()


# class ResizeNormalize:
#
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         self.size = size
#         self.interpolation = interpolation
#         self.toTensor = transforms.ToTensor()
#
#     def __call__(self, img):
#         img = img.resize(self.size, self.interpolation)
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         return img
#
#
# class CV2ResizeNormalize(object):
#
#     def __init__(self, size, interpolation=cv2.INTER_LINEAR):
#         self.size = size
#         self.interpolation = interpolation
#         self.toTensor = transforms.ToTensor()
#
#     def __call__(self, img):
#         img = cv2.resize(img, self.size, interpolation=self.interpolation)
#         # img = img.resize(self.size, self.interpolation)
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         return img
#
#
# # class RandomSequentialSampler(sampler.Sampler):
# #
# #     def __init__(self, data_source, batch_size):
# #         self.num_samples = len(data_source)
# #         self.batch_size = batch_size
# #
# #     def __iter__(self):
# #         n_batch = len(self) // self.batch_size
# #         tail = len(self) % self.batch_size
# #         index = torch.LongTensor(len(self)).fill_(0)
# #         for i in range(n_batch):
# #             random_start = random.randint(0, len(self) - self.batch_size)
# #             batch_index = random_start + torch.range(0, self.batch_size - 1)
# #             index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
# #         # deal with tail
# #         if tail:
# #             random_start = random.randint(0, len(self) - self.batch_size)
# #             tail_index = random_start + torch.range(0, tail - 1)
# #             index[(i + 1) * self.batch_size:] = tail_index
# #
# #         return iter(index)
# #
# #     def __len__(self):
# #         return self.num_samples
#
#
# class AlignCollate(object):
#
#     def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
#         self.imgH = imgH
#         self.imgW = imgW
#         self.keep_ratio = keep_ratio
#         self.min_ratio = min_ratio
#
#     def __call__(self, batch):
#         images, labels = zip(*batch)
#
#         imgH = self.imgH
#         imgW = self.imgW
#         # if self.keep_ratio:
#         #     ratios = []
#         #     for image in images:
#         #         w, h = image.size
#         #         ratios.append(w / float(h))
#         #     ratios.sort()
#         #     max_ratio = ratios[-1]
#         #     imgW = int(np.floor(max_ratio * imgH))
#         #     imgH = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
#
#         transform = ResizeNormalize((imgW, imgH))
#         images = [transform(image) for image in images]
#         images = torch.cat([t.unsqueeze(0) for t in images], 0)
#
#         return images, labels
