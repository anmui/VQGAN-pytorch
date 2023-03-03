import os
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import util
import utils
from Criterion import SetCriterion
from datasets import build_dataset
from discriminator import Discriminator
from lpips import LPIPS
from transformerr import Transformerr
from vqgan import VQGAN
from utils import load_data, weights_init
from fcn import FCN



def getTest(test_dataset_s,test_dataset_t,args):
    transformer = Transformerr(args).to(device=args.device)
    transformer.load_checkpoint(args.checkpoint_path_style);
    transformer.eval()
    with tqdm(range(len(test_dataset_s))) as pbar:
        for i, samples, style_images in zip(pbar,test_dataset_t, test_dataset_s):
            samples = samples.to(device=args.device)
            style_images = style_images.to(device=args.device)
            outputs = transformer.log_images(samples, style_images)
            with torch.no_grad():
                vutils.save_image(outputs, os.path.join("output/outputs", f"{i}.jpg"))
                vutils.save_image(outputs, os.path.join("output/style_images", f"{i}.jpg"))


