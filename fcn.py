import os
import torch
import torch.nn as nn
import torchvision
from matplotlib import transforms
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_ckpt_path(name, root):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
    return path


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.eval()
    def prepare_image(self,image1,image2):
        # 照片预处理，转为0-1之间，图像标准化
        img_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        img1_tensor = img_transform(image1).unsqueeze(0)
        img2_tensor = img_transform(image2).unsqueeze(0)
        return img1_tensor,img2_tensor

    def forward(self, real_x, fake_x):
        img1_tensor,img2_tensor = self.prepare_image(real_x,fake_x)
        output1 = self.fcn(img1_tensor)['out']
        output2 = self.fcn(img2_tensor)['out']
        return torch.abs(output2 - output1)