# DDPM PyTorch Implementation 

This repo contains a work-in-progress implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM) in PyTorch. 

## Usage 

```python
import ddpm_pytorch
from ddpm_pytorch import diffusion

diffuser = diffusion.DDPM(
    model=Unet(dim=64), 
    image_shape=(3, 32, 32), 
    trainloader=trainloader, 
    num_time_steps=1000, 
    loss='mse'
)

diffuser.train(num_epochs=100)

generated_image = diffuser.sample(
    model=diffuser.model, 
    num_time_steps=1000, 
    shape=(1,3,32,32)
)
```

## Installation

```
$ pip install git+https://github.com/rosikand/ddpm-pytorch.git
```
