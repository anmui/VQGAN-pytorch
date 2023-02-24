import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

import utils
from Criterion import SetCriterion
from discriminator import Discriminator
from lpips import LPIPS
from transformerr import Transformer
from vqgan import VQGAN
from utils import load_data, weights_init
from fcn import FCN



class TrainT:
    def __init__(self, args):
        weight_dict = {'loss_content': args.content_loss_coef, 'loss_style': args.style_loss_coef,
                       'loss_tv': args.tv_loss_coef}
        losses = ['content', 'style', 'tv']
        self.transformer = Transformer(args).to(device=args.device)
        criterion = SetCriterion(1, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        criterion.to(device=args.device)
        self.opt_t = self.configure_optimizers(args)
        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_t = torch.optim.Adam(
            list(self.transformer.transformer.parameters()) +
            list(self.transformer.tail.parameters()) +
            list(self.transformer.output_proj.parameters()) ,
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_t, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_loader_s,train_loader_t = utils.load_data_2(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_loader_t))) as pbar:
                for i, samples,style_images in zip(pbar, train_loader_t,train_loader_s):
                    samples = samples.to(device=args.device)
                    style_images = style_images.to(device=args.device)
                    outputs = self.transformer(samples, style_images)

                    loss_dict = self.criterion(outputs, samples, style_images)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = utils.reduce_dict(loss_dict)
                    loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                                  for k, v in loss_dict_reduced.items()}
                    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                                for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                    loss_value = losses_reduced_scaled.item()
                    self.opt_t.zero_grad()
                    losses.backward()
                    self.opt_t.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((samples[:4].add(1).mul(0.5)[:4], outputs.add(1).mul(0.5)[:4],style_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"1_{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        t_Loss=np.round(loss_value, 5)
                    )
                    pbar.update(0)
                torch.save(self.transformer.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=2, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path_s = [r"/media/lab/sdb/zzc/zhangdaqian",r"/media/lab/sdb/zzc/A"]
    args.dataset_path_t = [r"/media/lab/sdb/zzc/B"]
    train_vqgan = Transformer(args)



