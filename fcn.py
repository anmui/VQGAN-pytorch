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

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.eval()
    def forward(self, real_x, fake_x):
        #img1_tensor,img2_tensor = self.prepare_image(real_x,fake_x)
        output1 = self.fcn(real_x)['out']
        output2 = self.fcn(fake_x)['out']
        #return ((norm_tensor(output1)-norm_tensor(output2))**2).sum()

        return torch.mean(F.relu(torch.abs(output1 - output2)))