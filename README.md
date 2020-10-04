# BigGAN
Unofficial Pytorch implementation of BigGAN, proposed in [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) (Brock et al. 2018). Implementation for [Generative Adversarial Networks (GANs) Specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans) course material.

## Usage
1. Download the [ImageNet dataset](http://www.image-net.org/), into `data` directory and uncomment the ImageNet fields in `config.yml`. If you choose to run on STL10, you can skip this step since `torchvision` automatically downloads it.
2. All Python requirements can be found in `requirements.txt`. Support for Python>=3.7.
3. All defaults can be found in `config.yml` and are as per the configurations described in the original paper and code.

### Training
By default, all checkpoints will be stored in `logs/YYYY-MM-DD_hh_mm_ss`, but this can be edited via the `train_sr*.log_dir` field in the config file. If resuming from checkpoint, populate the `resume_checkpoint` field.

1. Run `python train.py` to train BigGAN.

### Inference
1. Edit the `resume_checkpoint` field in `config.yml` to reflect the desired checkpoint from training and run `python infer.py`. Use the `--labels` flag to specify which image classes to generate via comma-separated list of class labels. Defaults to `0,1,2,3,4`.
