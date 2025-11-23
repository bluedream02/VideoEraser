# MIT License

# Copyright (c) 2023 Hans Brouwer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import platform
import re
import warnings
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import torchvision

import numpy as np
import torch
from compel import Compel
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline, UNet3DConditionModel
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange

from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark
from utils.lora import inject_inferable_lora

def projection_matrix(E):
    """Calculate the projection matrix onto the subspace spanned by E."""   
    P = E @ torch.pinverse((E.T @ E).float()) @ E.T
    return P

def mask_to_onp(input_embeddings, p_emb, masked_input_subspace_projection, concept_subspace_projection, 
                alpha=0., max_length=77, debug=False):
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = ie.device
    (n_t, dim) = p_emb.shape   

    I_m_cs = torch.eye(dim).to(device) - cs
    dist_vec = I_m_cs @ p_emb.T
    dist_p_emb = torch.norm(dist_vec, dim=0)
        
    means = []
    
    for i in range(n_t):
        mean_without_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:])))
        means.append(mean_without_i)

    mean_dist = torch.tensor(means).to(device)
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float()
    inv_vector = (dist_p_emb >= (1. + alpha) * mean_dist).float()
    
    n_removed = n_t - rm_vector.sum()
    print(f"Among {n_t} tokens, we remove {int(n_removed)}.")
    
    ones_tensor = torch.ones(max_length).to(device)
    ones_tensor[1:n_t+1] = rm_vector
    ones_tensor = ones_tensor.unsqueeze(1)
        
    inverse_tensor = torch.ones(max_length).to(device)
    inverse_tensor[1:n_t+1] = inv_vector

    uncond_e, text_e = ie.chunk(2)
    text_e = text_e.squeeze()
    new_text_e = I_m_cs @ ms @ text_e.T
    new_text_e = new_text_e.T
    
    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e)
    new_embeddings = torch.concat([uncond_e, merged_text_e.unsqueeze(0)])
    return new_embeddings, ones_tensor, inverse_tensor, n_removed.item()

class CustomTextToVideoSDPipeline(TextToVideoSDPipeline):

    # Override the from_pretrained method to ensure compatibility
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    #     # Call the parent class's from_pretrained to ensure all parameters are properly handled
    #     return super(TextToVideoSDPipeline, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Add new class function
    def _new_encode_negative_prompt2(self, negative_prompt2, max_length, num_images_per_prompt, pooler_output=True):
        device = self._execution_device

        uncond_input = self.tokenizer(
            negative_prompt2,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        )
        if not pooler_output:
            uncond_embeddings = uncond_embeddings[0]
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            uncond_embeddings = uncond_embeddings.pooler_output
        
        return uncond_embeddings
    
    def _new_encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, 
                            prompt_ids=None, prompt_embeddings=None, token_mask=None, debug=False):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        detect_dict = {}
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        device = self._execution_device

        if prompt_embeddings is not None:
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            text_embeddings = self._encode_embeddings(
                prompt_ids,
                prompt_embeddings,
                attention_mask=attention_mask,
            )
            text_input_ids = prompt_ids
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            ################################################################################################
            # for null or mask_to_onp in tfg_type
            if token_mask is not None:
                mask_iids = torch.where(token_mask == 0, torch.zeros_like(token_mask), text_input_ids[0].to(device)).int()
                mask_iids = mask_iids[mask_iids != 0]
                tmp_ones = torch.ones_like(token_mask) * 49407
                tmp_ones[:len(mask_iids)] = mask_iids
                text_input_ids = tmp_ones.int()
                text_input_ids = text_input_ids[None, :]                            
            ################################################################################################

            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
        text_embeddings = text_embeddings[0]
        
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes




            concept_input = self.tokenizer(
                [concept],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            concept_embeddings = self.text_encoder(concept_input.input_ids.to(self.device))[0]

            # duplicate concept embeddings for each generation per prompt, using mps friendly method
            seq_len = concept_embeddings.shape[1]
            concept_embeddings = concept_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            concept_embeddings = concept_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
        


            text_embeddings = torch.cat([uncond_embeddings, text_embeddings,concept_embeddings])
        return text_embeddings, detect_dict, text_input_ids, text_inputs.attention_mask
    
    def _masked_encode_prompt(self, prompt):
        device = self._execution_device
        
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        n_real_tokens = untruncated_ids.shape[1] -2

        if untruncated_ids.shape[1] > self.tokenizer.model_max_length:
            untruncated_ids = untruncated_ids[:, :self.tokenizer.model_max_length]
            n_real_tokens = self.tokenizer.model_max_length -2
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        for i in range(n_real_tokens):
            masked_ids[i, i+1] = 0

        masked_embeddings = self.text_encoder(
            masked_ids.to(device),
            attention_mask=None,
        )
        return masked_embeddings.pooler_output
    


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet  # This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet")

    pipe = CustomTextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet.disable_gradient_checkpointing()
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()

    inject_inferable_lora(pipe, lora_path, r=lora_rank)

    return pipe


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        # initialize with random gaussian noise
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        # encode init_video to latents
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()


def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    latents: Tensor,
    init_weight: float,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    prompt_embeds: Optional[List[Tensor]],
    negative_prompt_embeds: Optional[List[Tensor]],
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
    rotate: bool,
    concept: str,
):
    device = pipe.device
    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size, _, num_frames, _, _ = latents.shape
    window_size = min(num_frames, window_size)

    prompt = prompt[0]

    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    negative_prompt2 = concept
    
    negative_prompt = ", ".join(negative_prompt2)



    text_embeddings, detect_dict, text_input_ids, attention_mask = pipe._new_encode_prompt(
            prompt, 1, do_classifier_free_guidance, negative_prompt, 
            None, None            
        )
    
    null_text_embeddings, _, _, _ = pipe._new_encode_prompt(
            prompt, 1, do_classifier_free_guidance, None, None, None
        )

    null_text_embeddings = null_text_embeddings.chunk(2)[0]  

    masked_embs = pipe._masked_encode_prompt(prompt)

    masked_project_matrix = projection_matrix(masked_embs.T) 

    neg2_text_embeddings = pipe._new_encode_negative_prompt2(negative_prompt2, 77, 1)
    project_matrix = projection_matrix(neg2_text_embeddings.T)
    
    rescaled_text_embeddings, sp_vector, inv_vector, n_removed = mask_to_onp(text_embeddings, masked_embs,
                                                                masked_project_matrix, 
                                                                project_matrix,
                                                                alpha=0.01,
                                                                debug=False)
    


    # set the scheduler to start at the correct timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_step = round(init_weight * len(pipe.scheduler.timesteps))
    timesteps = pipe.scheduler.timesteps[start_step:]
    if init_weight == 0:
        latents = torch.randn_like(latents)
    else:
        latents = pipe.scheduler.add_noise(
            original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
        )

    # manually track previous outputs for the scheduler as we continually change the section of video being diffused
    model_outputs = [None] * order

    if rotate:
        shifts = np.random.permutation(primes_up_to(window_size))
        total_shift = 0

    with pipe.progress_bar(total=len(timesteps) * num_frames // window_size) as progress:
        for i, t in enumerate(timesteps):
            progress.set_description(f"Diffusing timestep {t}...")

            if rotate:  # rotate latents by a random amount (so each timestep has different chunk borders)
                shift = shifts[i % len(shifts)]
                model_outputs = [None if pl is None else torch.roll(pl, shifts=shift, dims=2) for pl in model_outputs]
                latents = torch.roll(latents, shifts=shift, dims=2)
                total_shift += shift

            new_latents = torch.zeros_like(latents)
            new_outputs = torch.zeros_like(latents)

            for idx in range(0, num_frames, window_size):  # diffuse each chunk individually
                # update scheduler's previous outputs from our own cache
                pipe.scheduler.model_outputs = [model_outputs[(i - 1 - o) % order] for o in reversed(range(order))]
                pipe.scheduler.model_outputs = [
                    None if mo is None else mo[:, :, idx : idx + window_size, :, :].to(device)
                    for mo in pipe.scheduler.model_outputs
                ]
                pipe.scheduler.lower_order_nums = min(i, order)

                latents_window = latents[:, :, idx : idx + window_size, :, :].to(pipe.device)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_window] * 3) if do_classifier_free_guidance else latents_window
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=rescaled_text_embeddings).sample

                concept_momentum = None

                concept_guidance_scale: Optional[float] = 5000
                concept_warmup_steps: Optional[int] = 0
                concept_threshold: Optional[float] = 1.0
                concept_momentum_scale: Optional[float] = 0.5
                concept_mom_beta: Optional[float] = 0.7

                if do_classifier_free_guidance:


                    noise_pred_out = noise_pred.chunk(3)
                    noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]

                    noise_guidance = (noise_pred_text - noise_pred_uncond)

                    if True:
                        if concept_momentum is None:
                            concept_momentum = torch.zeros_like(noise_guidance)
                        noise_pred_concept = noise_pred_out[2]

                        scale = torch.clamp(
                            torch.abs((noise_pred_text - noise_pred_concept)) * concept_guidance_scale, max=1.)

                        concept_scale = torch.where(
                            (noise_pred_text - noise_pred_concept) >= concept_threshold,
                            torch.zeros_like(scale), scale)

                        noise_guidance_concept = torch.mul(
                            (noise_pred_concept - noise_pred_uncond), concept_scale)

                        noise_guidance_concept = noise_guidance_concept + concept_momentum_scale * concept_momentum

                        concept_momentum = concept_mom_beta * concept_momentum + (1 - concept_mom_beta) * noise_guidance_concept

                        if i >= concept_warmup_steps:
                            noise_guidance = noise_guidance - noise_guidance_concept

                    noise_pred = noise_pred_uncond + guidance_scale * noise_guidance


                # reshape latents for scheduler
                pipe.scheduler.model_outputs = [
                    None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                    for mo in pipe.scheduler.model_outputs
                ]
                latents_window = rearrange(latents_window, "b c f h w -> (b f) c h w")
                noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

                # compute the previous noisy sample x_t -> x_t-1
                latents_window = pipe.scheduler.step(noise_pred, t, latents_window).prev_sample

                # reshape latents back for UNet
                latents_window = rearrange(latents_window, "(b f) c h w -> b c f h w", b=batch_size)

                # write diffused latents to output
                new_latents[:, :, idx : idx + window_size, :, :] = latents_window.cpu()

                # store scheduler's internal output representation in our cache
                new_outputs[:, :, idx : idx + window_size, :, :] = rearrange(
                    pipe.scheduler.model_outputs[-1], "(b f) c h w -> b c f h w", b=batch_size
                )

                progress.update()

            # update our cache with the further denoised latents
            latents = new_latents
            model_outputs[i % order] = new_outputs

    if rotate:
        new_latents = torch.roll(new_latents, shifts=-total_shift, dims=2)

    return new_latents


@torch.inference_mode()
def inference(
    pipe,
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    window_size: Optional[int] = None,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    init_video: Optional[str] = None,
    init_weight: float = 0.5,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    loop: bool = False,
    seed: Optional[int] = None,
    concept=None,
):
    if seed is not None:
        torch.manual_seed(seed)

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        # pipe = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank)

        # prepare prompts
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds, negative_prompt_embeds = compel(prompt), compel(negative_prompt) if negative_prompt else None

        # prepare input latents
        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            init_video=init_video,
            vae_batch_size=vae_batch_size,
        )
        init_weight = init_weight if init_video is not None else 0  # ignore init_weight as there is no init_video!

        # run diffusion
        latents = diffuse(
            pipe=pipe,
            latents=init_latents,
            init_weight=init_weight,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            rotate=loop or window_size < num_frames,
            concept=concept
        )

        # decode latents to pixel space
        videos = decode(pipe, latents, vae_batch_size)

    return videos


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser(description="VideoEraser for ModelScope/ZeroScope: Generate videos with concept erasure")
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-d", "--data_path", type=str, default=None, help="Path to JSON data file (optional if using --prompt)")
    
    # Direct generation parameters (alternative to data_path)
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("-e", "--erased-concept", type=str, default=None, help="Concept to erase from generation")
    
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Directory to save output video to")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=256, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=256, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-WS", "--window-size", type=int, default=None, help="Number of frames to process at once (defaults to full sequence). When less than num_frames, a round robin diffusion process is used to denoise the full sequence iteratively one window at a time. Must be divide num_frames exactly!")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=25, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-i", "--init-video", type=str, default=None, help="Path to video to initialize diffusion from (will be resized to the specified num_frames, height, and width).")
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5, help="Strength of visual effect of init_video on the output (lower values adhere more closely to the text prompt, but have a less recognizable init_video).")
    parser.add_argument("-f", "--fps", type=int, default=12, help="FPS of output video")
    parser.add_argument("-d1", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-lP", "--lora_path", type=str, default="", help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lR", "--lora_rank", type=int, default=64, help="Size of the LoRA checkpoint's projection matrix (defaults to 64).")
    parser.add_argument("-rw", "--remove-watermark", action="store_true", help="Post-process the videos with LAMA to inpaint ModelScope's common watermarks.")
    parser.add_argument("-l", "--loop", action="store_true", help="Make the video loop (by rotating frame order during diffusion).")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    # Check if using direct command line or data file
    if args.prompt and args.erased_concept:
        # Direct command line mode
        print("Using direct command line arguments...")
        data = [["output", args.erased_concept, args.prompt, args.seed if args.seed else 42]]
    elif args.data_path:
        # Data file mode
        print(f"Loading data from {args.data_path}...")
        import json
        data = json.load(open(args.data_path))
    else:
        raise ValueError("Either provide --prompt and --erased-concept, or --data_path")

    import time
    start_time = time.time()


    if args.window_size is None:
        args.window_size = args.num_frames

    if args.init_video is not None:
        vr = decord.VideoReader(args.init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args.num_frames, args.height, args.width), mode="trilinear")
        args.init_video = init

    # =========================================
    # ============= sample videos =============
    # =========================================

    import json
    pipe = initialize_pipeline(model=args.model,
                            device=args.device, xformers=args.xformers,sdp=args.sdp,
                            lora_path=args.lora_path, lora_rank=args.lora_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model initialization time: {elapsed_time:.4f} seconds")
    import time
    start_time = time.time()

    for i, d in enumerate(data):

        prompt = d[2]
        global concept 
        concept = d[1]
        vid = d[0]
        seed = d[3]

        out_name = 'output_' + vid + '_erased_' + concept + '_seed_' + str(seed) + '.mp4'

        args.prompt = [prompt] * args.batch_size
        if args.negative_prompt is not None:
            args.negative_prompt = [args.negative_prompt] * args.batch_size

        videos = inference(
            pipe,
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            window_size=args.window_size,
            vae_batch_size=args.vae_batch_size,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            init_video=args.init_video,
            init_weight=args.init_weight,
            device=args.device,
            xformers=args.xformers,
            sdp=args.sdp,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            loop=args.loop,
            seed = d[3],
            concept=concept,
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Code execution time: {elapsed_time:.4f} seconds")
        




    # =========================================
    # ========= write outputs to file =========
    # =========================================
        output_dir = args.output#'/nas-hdd/shoubin/result/safegen/zeroscope_v2_576w+safree/'
        os.makedirs(args.output, exist_ok=True)

        for video in videos:
            if args.remove_watermark:
                print("Inpainting watermarks...")
                video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
                video = inpaint_watermark(video)
                video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

            else:
                video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

            video = video.byte().cpu().numpy()
            # video = [(frame * 255).astype(np.uint8) for frame in video]

            output_path = f"{output_dir}/{out_name}"
            torchvision.io.write_video(
                output_path, video, fps=16, video_codec="h264", options={"crf": "10"}
            )
            print(f"Video saved to: {output_path}")
        # break

