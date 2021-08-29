import os
from os.path import join
from os.path import split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image
import params
from deep_utils import split_extension


class CRNNDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):

        self.image_paths = [join(root, img_name) for img_name in os.listdir(root) if
                            split_extension(img_name)[-1].lower() in ['.jpg', '.png', '.jpeg']]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.image_paths[index]
        if params.nc == 1:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = self.get_label(img_path)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    @staticmethod
    def get_label(img_path):
        label = split_extension(split(img_path)[-1])[0]
        label = label.split('_')[-1].split('-')[0]
        return label


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


class ResizeNormalize:

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class CV2ResizeNormalize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        # img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


# class RandomSequentialSampler(sampler.Sampler):
#
#     def __init__(self, data_source, batch_size):
#         self.num_samples = len(data_source)
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         n_batch = len(self) // self.batch_size
#         tail = len(self) % self.batch_size
#         index = torch.LongTensor(len(self)).fill_(0)
#         for i in range(n_batch):
#             random_start = random.randint(0, len(self) - self.batch_size)
#             batch_index = random_start + torch.range(0, self.batch_size - 1)
#             index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
#         # deal with tail
#         if tail:
#             random_start = random.randint(0, len(self) - self.batch_size)
#             tail_index = random_start + torch.range(0, tail - 1)
#             index[(i + 1) * self.batch_size:] = tail_index
#
#         return iter(index)
#
#     def __len__(self):
#         return self.num_samples


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        # if self.keep_ratio:
        #     ratios = []
        #     for image in images:
        #         w, h = image.size
        #         ratios.append(w / float(h))
        #     ratios.sort()
        #     max_ratio = ratios[-1]
        #     imgW = int(np.floor(max_ratio * imgH))
        #     imgH = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = ResizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
