# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union, Dict, Any
from dataclasses import dataclass

import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FrameCondition:
    type: str  # 'controlnet', 'rgb', 'prompt'
    data: Any  # Specific data type depending on 'type'
    scale: float = 1.0  # Scale factor for the condition

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


def upscale_latent(latent_img, scale_factor, mode='nearest-exact'):
    """
    Upscales a latent image by a given scale factor using interpolation.

    Args:
    latent_img (torch.Tensor): The latent image to upscale.
    scale_factor (int): The factor by which to upscale the image.
    mode (str): The interpolation mode to use. Defaults to 'nearest-exact',  'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'

    Returns:
    torch.Tensor: The upscaled latent image.
    """
    # Assuming latent_img is a PyTorch tensor of shape [batch_size, channels, height, width]
    
    # Use interpolate function for upscaling
    if len(latent_img.shape) == 5:
        latent_img = latent_img.squeeze(0)
    upscaled_img = nn.functional.interpolate(latent_img, scale_factor=scale_factor, mode=mode)
    
    return upscaled_img.unsqueeze(0)

# Returns fraction that has denominator that is a power of 2
def ordered_halving(val, print_final=False):
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    final = as_int / (1 << 64)
    if print_final:
        print(f"$$$$ final: {final}")
    return final

def uniform_v2(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
    print_final: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    print(f'{num_frames=}, {context_size=}, {context_stride=}, {context_overlap=}')
    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)
    print(f'{context_stride=}')
    pad = int(round(num_frames * ordered_halving(step, print_final)))
    print(f'{pad=}')
    for context_step in 1 << np.arange(context_stride):
        j_initial = int(ordered_halving(step) * context_step) + pad
        print(j_initial,
            num_frames + pad - context_overlap,
            context_size * context_step - context_overlap)
        for j in range(
            j_initial,
            num_frames + pad - context_overlap,
            (context_size * context_step - context_overlap),
        ):
            if context_size * context_step > num_frames:
                # On the final context_step,
                # ensure no frame appears in the window twice
                yield [e % num_frames for e in range(j, j + num_frames, context_step)]
                continue
            j = j % num_frames
            if j > (j + context_size * context_step) % num_frames and not closed_loop:
                yield  [e for e in range(j, num_frames, context_step)]
                j_stop = (j + context_size * context_step) % num_frames
                # When  ((num_frames % (context_size - context_overlap)+context_overlap) % context_size != 0,
                # This can cause 'superflous' runs where all frames in
                # a context window have already been processed during
                # the first context window of this stride and step.
                # While the following commented if should prevent this,
                # I believe leaving it in is more correct as it maintains
                # the total conditional passes per frame over a large total steps
                # if j_stop > context_overlap:
                yield [e for e in range(0, j_stop, context_step)]
                continue
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


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
        controlnet: Optional[SparseControlNetModel] = None,
    ):
        super().__init__()
        self._adjust_scheduler_config(scheduler)
        self._adjust_unet_config(unet)
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
    def _adjust_scheduler_config(self, scheduler):
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                "The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                " should be set to 1 instead of {scheduler.config.steps_offset}. Please update the config."
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            scheduler._internal_dict = FrozenDict({**scheduler.config, "steps_offset": 1})

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                "The configuration file of this scheduler: {scheduler} has `clip_sample` set to True."
                " It should be False. Please update the config."
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            scheduler._internal_dict = FrozenDict({**scheduler.config, "clip_sample": False})

    def _adjust_unet_config(self, unet):
        if hasattr(unet.config, "_diffusers_version") and version.parse(unet.config._diffusers_version) < version.parse("0.9.0.dev0"):
            if unet.config.sample_size < 64:
                deprecation_message = (
                    "The configuration file of the unet has `sample_size` smaller than 64."
                    " It should be 64. Please update the config."
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                unet._internal_dict = FrozenDict({**unet.config, "sample_size": 64})

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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

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
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
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

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None,
                        alpha=None, freenoise=False, window_size=16, window_stride=4, freenoise_alpha=0.01):
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
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                if alpha is None:
                    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                    if freenoise:
                        print('Generate latents with FreeNoise')
                        for frame_index in range(window_size, video_length, window_stride):
                            list_index = list(
                                range(
                                    frame_index - window_size,
                                    frame_index + window_stride - window_size,
                                )
                            )
                            shape = (batch_size, num_channels_latents, len(list_index), height // self.vae_scale_factor, width // self.vae_scale_factor)
                            random.shuffle(list_index)
                            latents[
                                :, :, frame_index : frame_index + window_stride
                            ] = latents[:, :, list_index] + freenoise_alpha * torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                else:
                    shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    sigma = (1 + alpha ** 2) ** 0.5
                    latents = []
                    prev = None
                    
                    for i in range(video_length):
                        if i == 0:
                            latent = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                        else:
                            latent = prev + alpha * torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                            latent = latent / sigma
                            latent = latent - latent.mean()
                        latents.append(latent)
                        prev = latent
                    latents = torch.cat(latents, dim=2)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def interpolate_conditions(self, frame_conditions, video_length, latent_dim):
        # Prepare tensor to hold the interpolated conditions
        interpolated_conditions = torch.zeros((video_length, *latent_dim))

        # Sorted frames to ensure correct interpolation
        sorted_frames = sorted(frame_conditions.keys())
        for i in range(len(sorted_frames)):
            if i == len(sorted_frames) - 1:
                next_frame = video_length
                condition = frame_conditions[sorted_frames[i]].data
                interpolated_conditions[sorted_frames[i]:next_frame] = condition
            else:
                start_frame = sorted_frames[i]
                end_frame = sorted_frames[i + 1]
                start_condition = frame_conditions[start_frame].data
                end_condition = frame_conditions[end_frame].data
                # Linear interpolation
                for j in range(end_frame - start_frame):
                    t = j / (end_frame - start_frame)
                    interpolated_conditions[start_frame + j] = (1 - t) * start_condition + t * end_condition

        return interpolated_conditions
        
    @torch.no_grad()
    def prepare_control_cond(self,
                             latent_model_input,
                             text_embeddings,
                             t,
                             frame_conditions: Dict[int, FrameCondition],
                            #  controlnet_images, 
                            #  controlnet_image_index,
                            #  controlnet_conditioning_scale,
                             video_length,
                             device,
                             dtype):
        down_block_additional_residuals = mid_block_additional_residual = None
        if (getattr(self, "controlnet", None) is not None):

            controlnet_noisy_latents = latent_model_input
            controlnet_prompt_embeds = text_embeddings
            
            controlnet_cond = torch.zeros_like(latent_model_input).to(dtype).to(device)
            controlnet_conditioning_mask = torch.zeros_like(latent_model_input[:, :1, ...]).to(dtype).to(device)
            
            for frame_idx, condition in frame_conditions.items():
                assert condition.data.dim() == 5
                
                if condition.type == 'controlnet':
                    
                    controlnet_cond[:, :, frame_idx] = condition.data.to(dtype).to(device)
                    controlnet_conditioning_mask[:, :, frame_idx] = 1
                elif condition.type == 'rgb':
                    raise NotImplementedError("RGB conditioning not implemented.")
                elif condition.type == 'prompt':
                    raise NotImplementedError("Prompt conditioning not implemented.")

                # controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                # controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

            # Print shapes
            print(f"controlnet_noisy_latents.shape: {controlnet_noisy_latents.shape}")
            print(f"controlnet_prompt_embeds.shape: {controlnet_prompt_embeds.shape}")
            print(f"controlnet_cond.shape: {controlnet_cond.shape}")
            print(f"controlnet_conditioning_mask.shape: {controlnet_conditioning_mask.shape}")
            print(f"t: {t}, video_length: {video_length}")
            
            down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                controlnet_noisy_latents, t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=controlnet_cond,
                conditioning_mask=controlnet_conditioning_mask,
                conditioning_scale=condition.scale,
                guess_mode=False, return_dict=False,
            )
            
        return down_block_additional_residuals, mid_block_additional_residual

    def latent_upscale_sampling(
        self,
        latents: torch.FloatTensor,
        guidance_scale: float,
        text_embeddings: torch.FloatTensor,
        num_inference_steps: int,
        strength: float,
        upscale_factor: float,
        extra_step_kwargs: dict,
        generator = None,
        mode: str = 'bilinear',
        dtype=torch.float32,
        
        # support sliding window
        sliding_window: bool = False,
        context_length: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,
        
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        latents = upscale_latent(latents, upscale_factor, mode=mode)
        batch_size = latents.shape[0]
        video_length = latents.shape[2]
        
        # Prepare timesteps
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        
        timesteps, num_inference_steps = timesteps, num_inference_steps - t_start
        latent_timestep = timesteps[:1].repeat(batch_size)
        
        # Prepare noised latents
        shape = latents.shape
        noise = randn_tensor(shape, generator=generator, device=latents.device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(latents, noise, latent_timestep)
        latents = init_latents
        
        do_classifier_free_guidance = guidance_scale > 1.0
        down_block_additional_residuals, mid_block_additional_residual = None, None
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Sliding window support
                if sliding_window:
                    noise_pred = torch.zeros_like(latents, device=latents.device)
                    noise_pred = torch.cat([noise_pred] * 2) if do_classifier_free_guidance else noise_pred
                    
                    out_count_final = torch.zeros((1, 1, latents.shape[2], 1, 1), device=latents.device)
                    print(f"out_count_final.shape: {out_count_final.shape}")
                    for slide_frames in uniform_v2(step=i, num_steps=0, 
                                                   num_frames=video_length,
                                                   context_size=context_length,
                                                   context_stride=context_stride,
                                                   context_overlap=context_overlap,
                                                   closed_loop=False):
                        if len(slide_frames) == 0:
                            continue
                        out_count_final[:, :, slide_frames] += 1

                        latents_window = latents[:, :, slide_frames]
                        
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents_window] * 2) if do_classifier_free_guidance else latents_window
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        
                        print(f"step: {i}, frames: {slide_frames}")
                        print(f"latents_window.shape: {latents_window.shape}")
                        print(f"latent_model_input.shape: {latent_model_input.shape}")
                        print(f"text_embeddings.shape: {text_embeddings.shape}")
                        
                        noise_pred_window = self.unet(
                            latent_model_input, t, 
                            encoder_hidden_states=text_embeddings,
                            down_block_additional_residuals=down_block_additional_residuals,
                            mid_block_additional_residual=mid_block_additional_residual,
                        ).sample.to(dtype=latents.dtype)
                        
                        noise_pred[:, :, slide_frames] += noise_pred_window
                        
                    # Normalize latents
                    print(f"out_count_final: {out_count_final.squeeze().cpu().numpy()}")
                    noise_pred /= out_count_final
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t, 
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                    ).sample.to(dtype=latents.dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                        
        return latents
        
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: int,
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
        dtype=torch.float32,
        frame_conditions: Optional[Dict[int, FrameCondition]] = None,

        # autoregressive latent initialization
        latent_alpha: Optional[float] = None,
        # support freenoise
        freenoise: bool = False,
        freenoise_alpha: float = 0.01,
        # support sliding window
        sliding_window: bool = False,
        context_length: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,

        # support upscale
        apply_upscale: bool = False,
        strength: float = 0.6,
        upscale_factor: float = 1.5,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if latents is not None:
            batch_size = latents.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        ).to(dtype)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
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
            alpha=latent_alpha,
            freenoise=freenoise,
            freenoise_alpha=freenoise_alpha,
            window_size=context_length,
            window_stride=context_stride,
        ).to(dtype)
        latents_dtype = latents.dtype

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                down_block_additional_residuals, mid_block_additional_residual = None, None

                # Check if we need to apply conditional processing for this timestep
                if frame_conditions:
                    active_conditions = {frame_idx: cond for frame_idx, cond in frame_conditions.items() if frame_idx == i}
                    if active_conditions:
                        down_block_additional_residuals, mid_block_additional_residual = self.prepare_control_cond(
                            latent_model_input, text_embeddings, t, active_conditions, video_length, device, dtype
                        )

                # Predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual
                ).sample.to(dtype=latents_dtype)

                # Apply classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback and i % callback_steps == 0:
                        callback(i, t, latents)

        video = self.decode_latents(latents.to(torch.float32))

        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
