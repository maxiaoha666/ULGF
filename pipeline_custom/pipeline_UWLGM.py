'''
基本上仿照SD1.5 的 StableDiffusionPipeline
仅仅做unet部分的修改，原因：StableDiffusionPipeline中unet仅有text编码输入
'''

import inspect
from typing import Callable, List, Optional, Union

import torch
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.models import UNet2DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class UWLGMPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
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

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )

    # 如果输入是方形就是方形，不是就补成方形，记录原图像在方形中的位置信息
    def squarize(self, target_tensor, **bg_kwargs):
        batch, ch, height, width = target_tensor.size()
        if width == height:
            return target_tensor, (0, 0, width, height)
        elif width > height:
            top, bottom, left, right = (width - height) // 2, (width - height) // 2 + height, 0, width
            bg_tensor = self.scheduler.add_noise(
                torch.ones((batch, ch, width, width), device=target_tensor.device),
                torch.randn((batch, ch, width, width), device=target_tensor.device),
                bg_kwargs['timestep'].repeat(batch),
            )
            bg_tensor[:, :, top:bottom, left:right] = target_tensor
            return bg_tensor, (left, top, right, bottom)
        else:
            top, bottom, left, right = 0, height, (height - width) // 2, (height - width) // 2 + width
            bg_tensor = self.scheduler.add_noise(
                torch.ones((batch, ch, height, height), device=target_tensor.device),
                torch.randn((batch, ch, height, height), device=target_tensor.device),
                bg_kwargs['timestep'].repeat(batch),
            )
            bg_tensor[:, :, top:bottom, left:right] = target_tensor
            return bg_tensor, (left, top, right, bottom)

    #
    def trim_by_mask(self, target_tensor: torch.Tensor, mask_img: Image.Image):
        ## Get Bounding Box
        bbox = mask_img.getbbox()
        if bbox is None:
            raise Exception("Invalid mask image")

        ## Crop Image with bounding box
        target_tensor = target_tensor[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return target_tensor, bbox

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            **kwargs,
    ):
        # 判斷輸入為一組或者多組
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
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

        # get prompt text embeddings
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

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    " {type(prompt)}."
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

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size * num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
            latents_dtype = text_embeddings.dtype
            if latents is None:
                if self.device.type == "mps":
                    # randn does not exist on mps
                    latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                        self.device
                    )
                else:
                    latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents = latents.to(self.device)

            # set timesteps
            self.scheduler.set_timesteps(num_inference_steps)

            # Some schedulers like PNDM have timesteps as arrays
            # It's more optimized to move all timesteps to correct device beforehand
            timesteps_tensor = self.scheduler.timesteps.to(self.device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for i, t in enumerate(self.progress_bar(timesteps_tensor)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )

            if output_type == "pil":
                image = self.numpy_to_pil(image)

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)





