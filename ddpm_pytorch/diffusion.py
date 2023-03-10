"""
File: diffusion.py
------------------
Implements Denoising Diffusion Probabalistic Models. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,  # trainer 
        utils
    )
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import rsbox
from rsbox import ml, misc
import torchvision 
import os
import wandb


# ------------------------  (DDPM) ---------------------------------


class DDPM(experiment.Experiment):
    def __init__(
        self,
        model,
        image_shape,
        trainloader=None,
        num_time_steps=1000,
        loss='mse',
        optimizer=None,
        variance_schedule='linear',
        sample_every_n_epochs=1,
        save_weights_every_n_epochs=10,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        logger=None
    ): 
        self.model = model
        self.trainloader = trainloader
        assert loss == 'mse' or loss == "l1" or loss == 'huber' or loss == 'l2', 'Only mse, huber, l1, and l2 (mse) losses are supported. Please specify one of those as a string.'
        self.loss_str = loss
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        self.image_shape = image_shape
        self.num_time_steps = num_time_steps
        self.variance_schedule = self.get_variance_schedule(variance_schedule)
        self.precomputations = self.precompute_diffusion_variables()
        self.device = device
        self.model.to(self.device)
        self.epoch_num = 0
        self.logger = logger
        self.sample_every_n_epochs = sample_every_n_epochs  # int or None
        self.save_weights_every_n_epochs = save_weights_every_n_epochs  # int or None 


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = self.logger,
            verbose = True
        )


    def get_variance_schedule(self, variance_schedule_type):
        if variance_schedule_type == 'linear':
            return self.linear_beta_schedule(self.num_time_steps)
        elif variance_schedule_type == 'cosine':
            return self.cosine_beta_schedule(self.num_time_steps)
        elif variance_schedule_type == 'quadratic':
            return self.quadratic_beta_schedule(self.num_time_steps)
        elif variance_schedule_type == 'sigmoid':
            return self.sigmoid_beta_schedule(self.num_time_steps)
        else:
            raise ValueError('Only linear, cosine quadratic, and sigmoid variance schedules are supported. Please specify one of those as a string for variance_schedule.')


    def linear_beta_schedule(self, timesteps):
        # returns variance schedule 
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        citation: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def quadratic_beta_schedule(self, timesteps):
        """
        citation: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


    def sigmoid_beta_schedule(self, timesteps):
        """
        citation: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
        """
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    
    def precompute_diffusion_variables(self):
        alphas = 1. - self.variance_schedule
        alpha_prod = torch.cumprod(alphas, axis=0)  
        alpha_prod_prev = F.pad(alpha_prod[:-1], (1, 0), value=1.0)
        
        precomputations = {
            "alphas": alphas,
            "alpha_prod": alpha_prod,
            "alpha_prod_prev": alpha_prod_prev
        }

        return precomputations


    def forward_diffusion(self, x, timestep, noise_vector=None):
        # ddpm forward diffusion, returns noisy image, i.e. q_sample 
        if noise_vector is None:
            noise_vector = torch.randn_like(x).to(self.device)
        
        left = torch.sqrt(self.precomputations["alpha_prod"][timestep].to(self.device)) * x
        right = torch.sqrt(1 - self.precomputations["alpha_prod"][timestep].to(self.device)) * noise_vector
        final = left + right
        return final
    
    

    def visualize_diffusion(self, x, num_timesteps, n, plot=False):      
        # run diffusion 
        ims = []
        for i in range(0, num_timesteps, n):
            x_noisy = self.forward_diffusion(x, i)
            ims.append(x_noisy)
        
        ims = torch.cat(ims, axis=0)
        grid = torchvision.utils.make_grid(ims, nrow=5, padding=1, pad_value=1)
        
        if plot:
            ml.plot(grid)

        return grid
                
    
    def evaluate(self, batch):
        # returns loss for the trainer 

        batch_losses = []
        for x in batch:
            x = x.unsqueeze(0).to(self.device)
            noise_vector = torch.randn_like(x).to(self.device)
            sampled_timestep = torch.randint(0, self.num_time_steps, (1,)).to(self.device)
            noisy_x = self.forward_diffusion(x, sampled_timestep, noise_vector=noise_vector)
            pred_noise = self.model(noisy_x.to(self.device).float(), sampled_timestep.float()) 
            if self.loss_str == 'mse':
                loss_val = F.mse_loss(noise_vector.float(), pred_noise)
            elif self.loss_str == 'l1':
                loss_val = F.l1_loss(noise_vector.float(), pred_noise)
            elif self.loss_str == 'l2':
                loss_val = F.mse_loss(noise_vector.float(), pred_noise)
            elif self.loss_str == 'huber':
                loss_val = F.smooth_l1_loss(noise_vector.float(), pred_noise)
            batch_losses.append(loss_val)

        batch_loss = torch.mean(torch.stack(batch_losses))
        

        return batch_loss
        

    @torch.no_grad()
    def p_sample_step(self, xt, model, timestep):
        beta = self.variance_schedule[timestep]
        alpha_t = self.precomputations["alphas"][timestep]
        alpha_prod_t = self.precomputations["alpha_prod"][timestep]
        alpha_prod_prev_t = self.precomputations["alpha_prod_prev"][timestep]
        logits = model(xt, torch.tensor([timestep]).to(self.device))

        if timestep == 0:
            z = 0.0
        else:
            z = torch.randn_like(xt)

        # computation (line 4, algorithm 2)        
        alpha_comp = ((1 - alpha_t)/torch.sqrt(1 - alpha_prod_t))
        logits_alpha_mult = alpha_comp * logits
        inner_result = xt - logits_alpha_mult
        
        one_over_alpha_t_sqrt =  1/torch.sqrt(alpha_t)

        # posterior variance 
        sigma_t = beta * ((1 - alpha_prod_prev_t)/(1 - alpha_prod_t))

        final = (one_over_alpha_t_sqrt * inner_result) + (torch.sqrt(sigma_t) * z)

        return final 

    
    @torch.no_grad()
    def alt_p_sample_step(self, xt, model, timestep):
        # alternate p_sample_step, which uses the mu parameterization (equation 11)

        beta = self.variance_schedule[timestep]
        alpha_t = self.precomputations["alphas"][timestep]
        alpha_prod_t = self.precomputations["alpha_prod"][timestep]
        alpha_prod_prev_t = self.precomputations["alpha_prod_prev"][timestep]
        logits = model(xt, torch.tensor([timestep]).to(self.device))


        alpha_t_sqrt_recp = 1/torch.sqrt(alpha_t)
        model_mean = xt - ((beta / torch.sqrt(1 - alpha_prod_t)) * logits) 
        model_std = torch.sqrt(beta * ((1. - alpha_prod_prev_t)/(1. - alpha_prod_t)))

        if timestep == 0:
            z_noise = 0.0
        else:
            z_noise = torch.randn_like(xt)
        
        x_t_minus_1 = (alpha_t_sqrt_recp * model_mean) + (model_std * z_noise)

        return x_t_minus_1



    @torch.no_grad()
    def sample(self, model, num_time_steps, shape, view_every=100, save_name=misc.timestamp(), return_intermediates=False):
        # returns p_sample, i.e. the image after diffusion 
        # generates an image 
        # specify None to save_name to not save to file 
        
        model.to(self.device)

        # start with pure gaussian noise 
        xt = torch.randn(shape).to(self.device)

        saved_tensors = []

        for i in tqdm(reversed(range(0, num_time_steps)), desc='sampling loop', total=num_time_steps):
            # save and plot 
            if i % view_every == 0 or i == num_time_steps - 1:
                saved_tensors.append(xt.cpu())


                if self.wandb_logger is not None:
                    self.wandb_logger.log({save_name: wandb.Image(xt.cpu(), caption=f"step_{i}")})
                
    
            xt = self.alt_p_sample_step(xt, model, i)


        ims = torch.cat(saved_tensors, axis=0)
        grid = torchvision.utils.make_grid(ims, nrow=5, padding=1, pad_value=1)

        if save_name is not None:
            if not os.path.exists("saved_samples"):
                os.makedirs("saved_samples")

            save_path = f"saved_samples/sample_{save_name}.png"
            torchvision.utils.save_image(grid, save_path)
            print(f"Saved sample to {save_path}")
            

        if self.wandb_logger is not None:
            if self.epoch_num > 0:
                caption = "epoch_" + str(self.epoch_num)
            else:
                caption = f"sample_{save_name}"
            self.wandb_logger.log({"Generated sample": wandb.Image(grid, caption=caption)})


        if return_intermediates:
            return xt, saved_tensors, grid
        else:
            return xt


    def on_run_start(self):
        assert self.trainloader is not None, "Must provide a trainloader to train on before calling train()" 
        print(f"Training on device: {self.device}")


    def on_run_end(self):
        self.save_weights()


    def on_epoch_end(self):
        self.epoch_num += 1

        if self.sample_every_n_epochs is not None:
            save_path = "epoch_" + str(self.epoch_num) + "-" + misc.timestamp()
            self.sample(self.model, self.num_time_steps, (1, self.image_shape[-3], self.image_shape[-2], self.image_shape[-1]), view_every=200, save_name=save_path)

        if self.save_weights_every_n_epochs is not None:
            if self.epoch_num % self.save_weights_every_n_epochs == 0:
                if not os.path.exists("saved"):
                    os.makedirs("saved")
                save_path = "saved/epoch_" + str(self.epoch_num) + "-" + misc.timestamp()
                self.save_weights(save_path)
