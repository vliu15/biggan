# MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from datetime import datetime
import os

import yaml
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate

from modules.dataset import collate_fn
from modules.loss import BigGANLoss
from utils import weights_init, sample_z


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    return parser.parse_args()


def train(dataloaders, models, optimizers, train_config, device, start_epoch=0):
    ''' Train function for BigGAN '''
    # unpack modules
    train_dataloader, val_dataloader = dataloaders
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers

    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    loss = BigGANLoss(device=device)

    for epoch in range(start_epoch, train_config.epochs):

        # training epoch
        epoch_steps = 0
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        generator.train()
        discriminator.train()
        pbar = tqdm(train_dataloader, position=0, desc='train [G loss: -.-----][D loss: -.-----]')
        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            z = sample_z(generator.z_dim, x.shape[0], device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                g_loss, d_loss, x_fake = loss(generator, discriminator, x, y, z)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            epoch_steps += 1
            pbar.set_description(desc=f'train [G loss: {mean_g_loss/epoch_steps:.5f}][D loss: {mean_d_loss/epoch_steps:.5f}]')

        if epoch+1 % train_config.save_every == 0:
            print(f'Epoch {epoch}: saving checkpoint')
            torch.save({
                'g_state_dict': generator.state_dict(),
                'd_state_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(log_dir, f'epoch={epoch}.pt'))

        # validation epoch
        epoch_steps = 0
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        generator.eval()
        discriminator.eval()
        pbar = tqdm(val_dataloader, position=0, desc='val [G loss: -.-----][D loss: -.-----]')
        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            z = sample_z(generator.z_dim, x.shape[0], device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, x_fake = loss(generator, discriminator, x, y, z)

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            epoch_steps += 1
            pbar.set_description(desc=f'val [G loss: {mean_g_loss/epoch_steps:.5f}][D loss: {mean_d_loss/epoch_steps:.5f}]')


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = instantiate(config.generator).to(device).apply(weights_init)
    discriminator = instantiate(config.discriminator).to(device).apply(weights_init)

    g_optimizer = torch.optim.Adam(generator.parameters(), **config.g_optim)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), **config.d_optim)

    start_epoch = 0
    if config.resume_checkpoint is not None:
        state_dict = torch.load(config.resume_checkpoint)

        generator.load_state_dict(state_dict['g_model_dict'])
        discriminator.load_state_dict(state_dict['d_model_dict'])
        g_optimizer.load_state_dict(state_dict['g_optim_dict'])
        d_optimizer.load_state_dict(state_dict['d_optim_dict'])
        start_epoch = state_dict['epoch']
        print('Starting BigGAN training from checkpoint')
    else:
        print('Starting BigGAN training from random initialization')

    train_dataloader = torch.utils.data.DataLoader(
        instantiate(config.train_dataset),
        collate_fn=collate_fn,
        **config.train_dataloader,
    )
    val_dataloader = torch.utils.data.DataLoader(
        instantiate(config.val_dataset),
        collate_fn=collate_fn,
        **config.val_dataloader,
    )

    train(
        [train_dataloader, val_dataloader],
        [generator, discriminator],
        [g_optimizer, d_optimizer],
        config.train, device, start_epoch,
    )


if __name__ == '__main__':
    main()
