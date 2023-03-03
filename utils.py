import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, paths, size=None):
        self.size = size
        self.images = []
        for path in paths:
            self.images.extend([os.path.join(path, file) for file in os.listdir(path)])
        self._length = len(self.images)
        print(self._length)
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


def load_data_2(args):
    train_data_s = ImagePaths(args.dataset_path_s, size=256)
    train_loader_s = DataLoader(train_data_s, batch_size=args.batch_size, shuffle=False)
    train_data_t = ImagePaths(args.dataset_path_t, size=256)
    train_loader_t = DataLoader(train_data_t, batch_size=args.batch_size, shuffle=False)
    train_dataset_s, test_dataset_s = random_split(train_loader_s.dataset, [int(len(train_loader_s) * 0.8),
                                                                            len(train_loader_s) - int(
                                                                                len(train_loader_s) * 0.8)])
    train_dataset_t, test_dataset_t = random_split(train_loader_t.dataset, [int(len(train_loader_t) * 0.8),
                                                                            len(train_loader_t) - int(
                                                                                len(train_loader_t) * 0.8)])
    return train_loader_s, train_loader_t, test_dataset_s, test_dataset_t


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
