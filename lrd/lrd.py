from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch import autocast


class LRDiff(nn.Module):
    def __init__(self,channel,height,width,clip_tokenizer,clip,vae,unet,num_train_timesteps,scheduler,device):
        super().__init__()
        self.device=device
        self.clip_tokenizer=clip_tokenizer
        self.clip=clip
        self.vae=vae
        self.unet=unet
    
        self.channel=channel
        self.height=height
        self.width=width
        
        #schedular 
        self.num_train_timesteps=num_train_timesteps
        self.scheduler = scheduler
        
    
    @torch.no_grad()
    def token_context(self,prompt,layer_prompt):
        self.clip.to(self.device)

        tokens_unconditional = self.clip_tokenizer("", padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = self.clip(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

        tokens_conditional = self.clip_tokenizer(prompt, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = self.clip(tokens_conditional.input_ids.to(self.device)).last_hidden_state

        layer_tokens_conditional = self.clip_tokenizer(layer_prompt, padding="max_length", max_length=self.clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        layer_embedding_conditional = self.clip(layer_tokens_conditional.input_ids.to(self.device)).last_hidden_state

        self.clip.to("cpu")
        return embedding_unconditional, embedding_conditional, layer_embedding_conditional
    
    
    @torch.no_grad()
    def update_attention(self, topk, use_topk):
        def _attention(self, query, key, value, sequence_length, dim):
            batch_size_attention = query.shape[0]

            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )

            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
                attn_slice = (
                    torch.einsum("bid, bjd -> bij", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
                )
                attn_slice = attn_slice.softmax(dim=-1)

                if self._use_topk==True:
                    #find topk values
                    topk_value,topk_index = torch.topk(attn_slice,k=topk,dim=-1)
                    topk_value=topk_value[:,:,-1].unsqueeze(2)

                    #build atten_mask
                    attn_mask = ((attn_slice)>topk_value).to(attn_slice.dtype)
                    attn_slice = torch.einsum("bij,bij->bij",attn_slice,attn_mask)

                attn_slice = torch.einsum("bij, bjd -> bid", attn_slice, value[start_idx:end_idx])
                hidden_states[start_idx:end_idx] = attn_slice

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states
    
    
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name and "down_blocks" in name:
                module._use_topk=use_topk
                module._attention=_attention.__get__(module, type(module))
                
    
    @torch.no_grad()
    def calculate_delta(self, generator,layer_embedding_conditional,topk):
        self.update_attention(topk=topk,use_topk=True)
        
        batch_layers=layer_embedding_conditional.size()[0]
        noise_latent = torch.randn((batch_layers,self.channel,self.height//8,self.width//8), generator=generator, device=self.device)
        
        self.unet.to(self.device)
        with autocast("cuda"):
            delta=self.unet(noise_latent,self.num_train_timesteps, encoder_hidden_states=layer_embedding_conditional).sample
        self.unet.to("cpu")
        
        self.update_attention(topk=topk,use_topk=False)
        return delta
    
    
    @torch.no_grad()
    def calculate_eps(self, delta, delta_strength, masks, generator):
        self.vae.to(self.device)
        #vae decoder range
        delta=delta/0.18215
        #decode to real space images, return real images, and eps
        with autocast("cuda"):
            delta=self.vae.decode(delta.to(self.vae.dtype)).sample
            delta_image=delta.detach()
            delta=torch.mean(delta,dim=(2,3))
            delta=delta_strength*delta
            eps=torch.einsum("bc,bcij->bcij",delta, 2*masks-1)
            
            #normal distributation
            eps_latent=self.vae.encode(eps).latent_dist.sample(generator=generator)*0.18215
        self.vae.to("cpu")
        return delta_image, eps, eps_latent
    
    
    @torch.no_grad()
    def sample(self,
               steps,
               t_start,
               t0,
               embedding_unconditional,
               embedding_conditional,
               layer_embedding_conditional,
               eps_latent,
               masks_weight,
               gamma,
               guidance_scale,
               generator):
        
        self.unet.to(self.device)
        self.vae.to(self.device)
        
        init_latent = torch.zeros((1,self.channel,self.height//8,self.width//8), device=self.device)
        self.scheduler.set_timesteps(steps)
        noise_latent = torch.randn((1,self.channel,self.height//8,self.width//8), generator=generator, device=self.device)
        latent = self.scheduler.add_noise(init_latent, noise_latent, t_start).to(self.device)

        
        with autocast("cuda"):
            timesteps = self.scheduler.timesteps[t_start:]
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i
                sigma = self.scheduler.sigmas[t_index]

                latent_model_input = latent
                latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(self.unet.dtype)
                
                #Predict the unconditional noise residual
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                if t>t0:
                    noise_pred_layers = self.unet(latent_model_input+eps_latent, t, encoder_hidden_states=layer_embedding_conditional).sample
                    phy_t=torch.sum(torch.einsum('n,nchw->nchw',masks_weight,noise_pred_layers),dim=0)
                    noise_pred_cond=gamma*phy_t+(1-gamma)*noise_pred_uncond

                #perform guidance
                #guidance_scale=7.5,
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = self.scheduler.step(noise_pred, t_index, latent).prev_sample

            #scale and decode the image latents with vae
            latent = latent / 0.18215
            image = self.vae.decode(latent.to(self.vae.dtype)).sample
    
        self.vae.to("cpu")
        self.unet.to("cpu")
        return image