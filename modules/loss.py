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

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigGANLoss(nn.Module):
    ''' Implements BigGAN forward pass and composite loss functions '''

    def __init__(self, beta=0.0001, device='cuda'):
        super().__init__()
        self.beta = beta
        self.device = device

    def adv_loss(self, pred, is_real):
        target = torch.zeros_like(pred) if is_real else torch.ones_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)

    def orthogonal_regularization(self, model):
        loss = 0.0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                w = m.weight.flatten(1)
                loss += torch.norm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=self.device)))
        return loss

    def forward(self, generator, discriminator, x, y, z):
        ''' Performs forward pass and returns total losses for G and D '''
        x_fake = generator(z, y)

        fake_preds_for_g = discriminator(x_fake, y=y)
        fake_preds_for_d = discriminator(x_fake.detach(), y=y)
        real_preds_for_d = discriminator(x.detach(), y=y)

        g_loss = (
            self.adv_loss(fake_preds_for_g, False) + \
            self.beta * self.orthogonal_regularization(generator)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False) + \
                self.beta * self.orthogonal_regularization(discriminator)
        )

        return g_loss, d_loss, x_fake
