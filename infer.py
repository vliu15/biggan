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

import yaml
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from utils import sample_z, show_tensor_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-l', '--labels', type=str, default='0,1,2,3,4')
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = [int(l) for l in args.labels.split(',')]

    generator = instantiate(config.generator)
    generator.load_statei_dict(torch.load(config.resume_checkpoint)['g_state_dict'])

    for y in range(labels):
        y = torch.tensor((y), device=device)
        z = sample_z(generator.z_dim, 1, device)
        x_fake = generator(z, y)

        show_tensor_images(x_fake)


if __name__ == '__main__':
    main()
