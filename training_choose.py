import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from choosenet import Choose

class TrainChoose:
    def __init__(self, args):
        self.model = Choose(args).to(device=args.device)
        self.opt_ch = self.configure_optimizers(args)
        self.prepare_training()
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.train(args)
    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_ch = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.choose.parameters()) +
            list(self.model.quant_conv.parameters()) ,
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )

        return opt_ch

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.model(imgs)
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    choose_loss=q_loss+perceptual_loss

                    self.opt_ch.zero_grad()
                    choose_loss.backward(retain_graph=True)

                    self.opt_ch.step()

                    pbar.set_postfix(Ch_Loss=np.round(choose_loss.cpu().detach().numpy().item(), 5))
                    pbar.update(0)
            sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs, os.path.join("results", f"Choose{epoch}.jpg"), nrow=4)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"choose_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = r"C:\Users\dome\datasets\flowers"

    train_choose = TrainChoose(args)



