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
    transformer.load_checkpoint(args.checkpoint_path_transformer)
    transformer.eval()
    with tqdm(range(len(test_dataset_t))) as pbar:
        for i, samples, style_images in zip(pbar,test_dataset_t, test_dataset_s):
            samples = samples.to(device=args.device)
            style_images = style_images.to(device=args.device)
            outputs = transformer.log_images(style_images,samples)
            samples=samples[:4].add(1).mul(0.5)[:4]
            outputs=outputs.add(1).mul(0.5)[:4]
            style_images=style_images.add(1).mul(0.5)[:4]
            with torch.no_grad():
                vutils.save_image(outputs, os.path.join("output/outputs_fix", f"{i}.jpg"))
                vutils.save_image(style_images, os.path.join("output/style_images_fix", f"{i}.jpg"))
                vutils.save_image(samples, os.path.join("output/content_fix", f"{i}.jpg"))
def getTest(test_dataset_s,test_dataset_t,args,transformer):
    with tqdm(range(len(test_dataset_t))) as pbar:
        for i, samples, style_images in zip(pbar,test_dataset_t, test_dataset_s):
            samples = samples.to(device=args.device)
            style_images = style_images.to(device=args.device)
            outputs = transformer.log_images(style_images,samples)
            #samples,_,_=transformer.vqgan_t(samples)
            #style_images, _, _ = transformer.vqgan_s(style_images)
            samples=samples[:4].add(1).mul(0.5)[:4]
            outputs=outputs.add(1).mul(0.5)[:4]
            style_images=style_images.add(1).mul(0.5)[:4]
            with torch.no_grad():
                vutils.save_image(outputs, os.path.join("output/outputs", f"{i}.jpg"))
                vutils.save_image(style_images, os.path.join("output/style_images", f"{i}.jpg"))
                vutils.save_image(samples, os.path.join("output/content", f"{i}.jpg"))


