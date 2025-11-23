

enable_safety_guidance = True
enable_encoder_safety_guidance = True

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

import os
import matplotlib.pyplot as plt
import numpy as np

logger = logging.get_logger(__name__)

def mask_to_onp(input_embeddings, p_emb, masked_input_subspace_projection, concept_subspace_projection, 
                alpha=0., max_length=77, debug=False):
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection

    device = ie.device
    (n_t, dim) = p_emb.shape   

    I_m_cs = torch.eye(dim).to(device) - cs
    dist_vec = I_m_cs.bfloat16() @ p_emb.T.bfloat16()

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
    new_text_e = I_m_cs.bfloat16() @ ms.bfloat16() @ text_e.T.bfloat16()
    new_text_e = new_text_e.T

    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e)
    new_embeddings = torch.concat([uncond_e, merged_text_e.unsqueeze(0)])

    return new_embeddings, ones_tensor, inverse_tensor, n_removed.item()

def projection_matrix(E):

    P = E.float() @ torch.pinverse((E.T.float() @ E.float()).float()) @ E.T.float()

    return P

def efficient_projection_matrix(E):
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    rank_threshold = 1e-5  
    low_rank_indices = S > rank_threshold
    E_reduced = U[:, low_rank_indices] @ torch.diag(S[low_rank_indices])
    P = E_reduced @ torch.pinverse(E_reduced.T @ E_reduced) @ E_reduced.T
    return P

def dynamic_alpha(distances, base_alpha=0.01, beta=0.1):

    std_dev = torch.std(distances)
    mean = torch.mean(distances)
    dynamic_alpha = base_alpha + beta * (std_dev / mean)
    return dynamic_alpha

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,

    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

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

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return uncond_embeddings, text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _masked_encode_prompt(self, prompt):
        device = self._execution_device

        print('prompt', len(prompt.split()))

        untruncated_ids = self.tokenizer(
            prompt,
            padding="longest",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids

        print('untruncated_ids', untruncated_ids.shape)

        n_real_tokens = untruncated_ids.shape[1] -2

        if untruncated_ids.shape[1] > 77:
            untruncated_ids = untruncated_ids[:, :77]
            n_real_tokens = 77 -2
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        for i in range(n_real_tokens):
            masked_ids[i, i+1] = 0

        masked_embeddings = self.text_encoder(
            masked_ids.to(device),
            attention_mask=None,
        ).last_hidden_state

        masked_embeddings = masked_embeddings.mean(dim=1)

        return masked_embeddings

    def _new_encode_negative_prompt2(self, negative_prompt2, max_length, num_images_per_prompt, pooler_output=True):
        device = self._execution_device

        uncond_input = self.tokenizer(
            negative_prompt2,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
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
            uncond_embeddings = uncond_embeddings.last_hidden_state[:,0,:]

        return uncond_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        controlnet_images: torch.FloatTensor = None,
        controlnet_image_index: list = [0],
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,

        concept_guidance_scale: Optional[float] = 5000,
        concept_warmup_steps: Optional[int] = 0,
        concept_threshold: Optional[float] = 1.0,
        concept_momentum_scale: Optional[float] = 0.5,
        concept_mom_beta: Optional[float] = 0.7,

        concept: Optional[str] = 'Van Gogh',

        **kwargs,
    ):

        import time
        start_time = time.time()

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        if enable_safety_guidance == False:
            uncond_embeddings, text_embeddings = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            if enable_encoder_safety_guidance == False:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            else: 
                if isinstance(prompt, list):
                    prompt_str = " ".join(prompt)
                else:
                    prompt_str = prompt

                masked_embs = self._masked_encode_prompt(prompt_str)
                masked_project_matrix = efficient_projection_matrix(masked_embs.T)

                neg2_text_embeddings = self._new_encode_negative_prompt2(concept, 77, 1)

                project_matrix = efficient_projection_matrix(neg2_text_embeddings.T)

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                rescaled_text_embeddings, sp_vector, inv_vector, n_removed = mask_to_onp(text_embeddings, masked_embs,
                                                    masked_project_matrix, 
                                                    project_matrix,
                                                    alpha=0.1,
                                                    debug=False)

                text_embeddings = rescaled_text_embeddings

        else:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
            text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
            text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

            do_classifier_free_guidance = guidance_scale > 1.0
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
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
                uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

                if enable_encoder_safety_guidance:
                    if isinstance(prompt, list):
                        prompt_str = " ".join(prompt)
                    else:
                        prompt_str = prompt

                    masked_embs = self._masked_encode_prompt(prompt_str)
                    masked_project_matrix = projection_matrix(masked_embs.T) 

                    neg2_text_embeddings = self._new_encode_negative_prompt2(concept, 77, 1)

                    project_matrix = projection_matrix(neg2_text_embeddings.T)

                if True:
                    concept_input = self.tokenizer(
                        [concept],
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    concept_embeddings = self.text_encoder(concept_input.input_ids.to(self.device))[0]

                    seq_len = concept_embeddings.shape[1]
                    concept_embeddings = concept_embeddings.repeat(batch_size, num_videos_per_prompt, 1)
                    concept_embeddings = concept_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, concept_embeddings])

                    if enable_encoder_safety_guidance:

                        rescaled_text_embeddings, sp_vector, inv_vector, n_removed = mask_to_onp(text_embeddings, masked_embs,
                                                                            masked_project_matrix, 
                                                                            project_matrix,
                                                                            alpha=0.00001,
                                                                            debug=False)

                        if do_classifier_free_guidance:
                            text_embeddings = rescaled_text_embeddings

                else:
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Code execution time: {elapsed_time:.4f} seconds")
        import time
        start_time = time.time()

        lambda_temporal = 0.1
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            concept_momentum = None

            noise_pred_text1 = None
            noise_pred_concept1 = None       

            for i, t in enumerate(timesteps):

                if enable_safety_guidance == False:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                else:
                    latent_model_input = torch.cat([latents] * (3)) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None

                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    assert controlnet_images.dim() == 5

                    controlnet_noisy_latents = latent_model_input
                    controlnet_text_embeddings = text_embeddings

                    controlnet_images = controlnet_images.to(latents.device)

                    controlnet_cond_shape    = list(controlnet_images.shape)
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                    controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                    assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                    controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t,
                        encoder_hidden_states=controlnet_text_embeddings,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False, return_dict=False,
                    )

                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals = down_block_additional_residuals,
                    mid_block_additional_residual   = mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)

                if do_classifier_free_guidance:
                    if enable_safety_guidance==False:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_out = noise_pred.chunk(3)
                        noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]

                        noise_guidance = (noise_pred_text - noise_pred_uncond)

                        if concept_momentum is None:
                            concept_momentum = torch.zeros_like(noise_guidance)
                        noise_pred_concept = noise_pred_out[2]

                        step_consistency=0

                        frame2frame=1
                        if step_consistency==1:
                            if frame2frame==1:
                                dynamic_concept_scale = concept_guidance_scale * (t / timesteps[-1])
                                scale = torch.clamp(
                                    torch.abs((noise_pred_text - noise_pred_concept)) * dynamic_concept_scale, max=1.
                                )
                            else:
                                if noise_pred_text1== None:
                                    pass

                                noise_pred_text1=noise_pred_text
                                noise_pred_concept1 =noise_pred_concept+torch.randn_like(noise_pred_concept) *100

                                dynamic_concept_scale = concept_guidance_scale * (t / timesteps[-1])
                                scale = torch.clamp(
                                    torch.abs((noise_pred_text1 - noise_pred_concept1)) * dynamic_concept_scale, max=1.
                                )
                        else:
                            dynamic_concept_scale = concept_guidance_scale
                            scale = torch.clamp(
                                torch.abs((noise_pred_text - noise_pred_concept)) * dynamic_concept_scale, max=2.
                            )

                        concept_scale = torch.where(
                            (noise_pred_text - noise_pred_concept) >= concept_threshold,
                            torch.zeros_like(scale), scale)

                        noise_guidance_concept = torch.mul(
                            (noise_pred_concept - noise_pred_uncond), concept_scale)

                        noise_guidance_concept = noise_guidance_concept + concept_momentum_scale * concept_momentum

                        if frame2frame==1:
                            if i > 0:
                                temporal_loss = torch.mean((latents - prev_latents) ** 2)
                                noise_guidance_concept += lambda_temporal * temporal_loss
                        else:
                            pass

                        concept_momentum = concept_mom_beta * concept_momentum + (1 - concept_mom_beta) * noise_guidance_concept.mean(dim=2, keepdim=True)

                        if i >= concept_warmup_steps:
                            noise_guidance = noise_guidance - noise_guidance_concept

                        noise_pred = noise_pred_uncond + guidance_scale * noise_guidance

                prev_latents = latents.clone()

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Code execution time: {elapsed_time:.4f} seconds")

        video = self.decode_latents(latents)

        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
