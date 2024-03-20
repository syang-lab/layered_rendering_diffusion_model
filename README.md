# Layered Rendering Diffusion Model
Unofficial implementation of the paper "Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis". 

You can find the implementation in this Colab notebook:
<a href="https://drive.google.com/file/d/1KcNvrjh7k5G4FFbzeMfdGruA-o0Y4XZB/view?usp=share_link">layer rendering diffusion model </a>

## Objective of the Model
Control the layout of the images generated from difffuse model, without retraining or find tunning and make better sementic alignment.

## Evaluation Matrix
CLIP score: text image sementic alignment
IOU: layout

## Diffuse Model
The diffuse model consists of two distinct processes: forward and reverse (generation). Understanding the diffuse process from the physical perspective reveals that the forward process mimics the diffusion of pollen particles in water. During this phase, the diffuse model consistently introduces random noise into the system. In contrast, the reverse (generation) process is designed to entirely reverse the diffusion process, allowing for the collection of pollen. This reversal entails retracing the movements or distribution of diffusing particles in the opposite direction, ultimately restoring the initial configuration.

<figure>
  <img src="Fig1-DDPM.png">
  <figcaption>Figure 1. The directed graphical model of DDPM (Ho et al., 2020). </figcaption>
</figure>


## Latent Diffuse Model
Latent diffuse model can further reduced the time of forward and reverse process though performing the diffuse in the latent space without reducing the synthesis quality (Rombach et al., 2022). The architecture of latent diffuse model is shown in Figure 2. The latent diffuse model include two stages, the first stage contains a VAE (Razavi et al., 2019) or VQGAN  (Esser et al., 2021) model. The encoder \\(\varepsilon\\) encoded \\(x\\) into the latent space \\(z\\), the decoder \\(D\\) decode \\(z\\) into the image space. In the second stage, forward and reverse diffusion happens in the latent space \\(z\\),  hence reducing the training and inference time. The conditions are added to the diffusion model after embedded using encoder \\(\tau_{\theta}\\), the encoded conditions are query in the cross-attention layers of the modified Unet \\(\epsilon_{\theta}\\) model.

<figure>
  <img src="Fig3-LD.png">
  <figcaption>Figure 2. The architecture of latent diffuse model (Rombach et al.,2022) </figcaption>
</figure>

## Layered Rendering Diffusion Model
<figure>
  <img src="Fig3.png">
</figure>
The algorithm of layer rendering model is shown below:
<figure>
  <img src="Fig-algorithm.png">
</figure>











