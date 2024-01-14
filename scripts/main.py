from PIL import Image
import argparse

import torch
from transformers import CLIPModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import LMSDiscreteScheduler

from ..utils import load_mask, calculate_mask_weight, display_image
from ..lrd import LRDiff

def main(args):
    #load clip model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model
    #load diffusion model
    auth_token = "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"
    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
    #load vae model 
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)

    seed = args.seed
    generator = torch.cuda.manual_seed(seed)

    #load masks: stack mask_files 
    masks=load_mask(args.mask_files)
    masks_weight=calculate_mask_weight(masks)
    
    channel=args.channel
    height=args.height
    width=args.width
    num_train_timesteps=args.num_train_timesteps
    scheduler=LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=num_train_timesteps)
    device=args.device
    lrdiff=LRDiff(channel,height,width,clip_tokenizer,clip,vae,unet,num_train_timesteps,scheduler,device)

    #1.calculate text encoder
    prompt=args.prompt
    layer_prompt =args.layer_prompt
    embedding_unconditional, embedding_conditional, layer_embedding_conditional = lrdiff.token_context(prompt,layer_prompt)

    #2.calculate delta
    topk=args.topk
    delta=lrdiff.calculate_delta(generator,layer_embedding_conditional,topk)

    #3.calculate eps
    delta_strength=args.delta_strength
    masks=masks.to(device)
    delta_image, eps, eps_latent=lrdiff.calculate_eps(delta, delta_strength, masks, generator)

    #4.calculate 
    steps=args.steps
    t_start=args.t_start
    t0=args.t0
    gamma=args.gamma
    guidance_scale=args.guidance_scale
    masks_weight=masks_weight.to(device)

    out_image=lrdiff.sample(steps,
                            t_start,
                            t0,
                            embedding_unconditional,
                            embedding_conditional,
                            layer_embedding_conditional,
                            eps_latent,
                            masks_weight,
                            gamma,
                            guidance_scale,
                            generator)

    #5.display generated image
    image=display_image(out_image)
    
    #6.display masks
    masks_image=display_image(torch.mean(eps.detach(),dim=0,keepdim=True))

    #6.display mask overlapped image
    Image.blend(Image.fromarray(masks_image),Image.fromarray(image),0.2)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2483964025, type=int, required=True)
    parser.add_argument('--mask-files',default=["mask1.png","mask2.png"],nargs='+',type=str,required=True)
    
    parser.add_argument('--channel',default=4, type=int,required=True)
    parser.add_argument('--height',default=512, type=int,required=True)
    parser.add_argument('--width',default=512, type=int,required=True)

    parser.add_argument('--num-train-timesteps',default=1000, type=int,required=True)

    parser.add_argument('--device',default="cuda", type=str,required=True)

    parser.add_argument('--prompt',default="Iron-man and a green Hulk are standing on the ruins", type=str,required=True)
    parser.add_argument('--layer-prompt',default=["Iron-man", "a green Hulk"],nargs='+',type=str,required=True)

    parser.add_argument('--topk',default=10, type=int,required=True)
    parser.add_argument('--delta-strength',default=1.0, type=float,required=True)

    parser.add_argument('--steps',default=250, type=int,required=True)
    parser.add_argument('--t-start',default=0, type=int,required=True)
    parser.add_argument('--t0',default=900, type=int,required=True)
    parser.add_argument('--gamma',default=0.2, type=float,required=True)
    parser.add_argument('--guidance-scale',default=7.5, type=float,required=True)

    args = parser.parse_args()
        
    main(args)




    
    