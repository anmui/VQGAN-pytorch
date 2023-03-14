import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm
from lpips import norm_tensor,spatial_average
import torch.nn.functional as F
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.eval()
    def forward(self, real_x, fake_x):
        #img1_tensor,img2_tensor = self.prepare_image(real_x,fake_x)
        output1 = self.fcn(real_x)['out']
        output2 = self.fcn(fake_x)['out']
        print(output1)
        #
        # real_fake_images = torch.cat((output1[:4].add(1).mul(0.5)[:4], output2.add(1).mul(0.5)[:4]))
        # vutils.save_image(real_fake_images, os.path.join("results", f"B.jpg"), nrow=4)
        om = torch.argmax(output1.squeeze(), dim=0).detach().cpu().numpy()
        print(om)
        rgb = self.decode_segmap(om)
        plt.imshow(rgb)
        plt.show()
        #return ((norm_tensor(output1)-norm_tensor(output2))**2).sum()
        return torch.mean(F.relu(torch.abs(output1 - output2)))

    def decode_segmap(self,image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

