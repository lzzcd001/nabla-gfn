import os
from collections import defaultdict
import contextlib
import datetime
import time
import wandb
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import copy
import pickle, gzip

import math

import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverSinglestepScheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available

from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import lib.reward_func.prompts
import lib.reward_func.rewards
from lib.diffusion.sample_trajectory import sample_trajectory
from lib.diffusion.inference_step import inference_step, predict_clean, get_alpha_prod_t


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from torch.nn.attention import SDPBackend, sdpa_kernel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_integer("seed", 0, "Seed.")

def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main(args):
    train()

def setup(local_rank, is_local_main_process):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    config = FLAGS.config

    config.gpu_type = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    if is_local_main_process:
        logger.info(f"GPU type: {config.gpu_type}")

    # config.config_name = f"{FLAGS.config}"
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    else:
        config.seed = 0

    if config.model.no_flow:
        config.model.reverse_loss_scale = 0.0

    wandb_name = f"{config.experiment.reward_fn.split('_')[0]}_{FLAGS.exp_name}_seed{config.seed}"


    if config.logging.use_wandb:
        wandb_key = config.logging.wandb_key
        wandb.login(key=wandb_key)
        wandb.init(project=config.logging.proj_name, name=wandb_name, config=config.to_dict(),
           dir=config.logging.wandb_dir,
           save_code=True, mode="online" if is_local_main_process else "disabled")

    os.makedirs(config.saving.output_dir, exist_ok=True)

    if is_local_main_process:
        logger.info(f"\n{config}")
    set_seed(config.seed)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision, torch_dtype=weight_dtype,
    )
    scheduler_config = {}

    scheduler_config.update(pipeline.scheduler.config)
    if config.sampling.scheduler == 'DPM-solver':
        if is_local_main_process:
            logger.info("Using SDE DPM-solver (1st order)")
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(scheduler_config)
        pipeline.scheduler.config.algorithm_type = "sde-dpmsolver++"  # Switch to SDE mode
        ### Essential; otherwise the score function at the final step is of infinite magnitude
        pipeline.scheduler.config.final_sigmas_type = 'sigma_min'
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    else:
        pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)
    pipeline.scheduler.set_timesteps(config.sampling.num_steps, device=device)  # set_timesteps(): 1000 steps -> 50 steps

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    unet = pipeline.unet
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        param.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    unet_lora_config = LoraConfig(
        r=config.model.lora_rank, lora_alpha=config.model.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config, adapter_name="pf") ## LoRA

    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if is_local_main_process:
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        unet.enable_xformers_memory_efficient_attention()
        if is_local_main_process:
            logger.info("xformers is enabled for memory efficient attention")
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    if not config.model.no_flow:
        res_logflowscore_model = UNet2DConditionModel(
            in_channels=4, block_out_channels=config.model.flow_channel_width,
            layers_per_block=config.model.flow_layers_per_block, cross_attention_dim=pipeline.text_encoder.config.hidden_size
        )

        ### Zero initilaization
        try:
            res_logflowscore_model.conv_out.bias.data *= 0.0
        except:
            pass
    else:
        res_logflowscore_model = None

    unet.set_adapter("pf")
    if config.training.mixed_precision in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
        if not config.model.no_flow:
            cast_training_params(res_logflowscore_model, dtype=torch.float32)

    pf_params = [param for name, param in unet.named_parameters() if '.pf.' in name]
    if not config.model.no_flow:
        flow_params = filter(lambda p: p.requires_grad, res_logflowscore_model.parameters())

    if config.training.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler(
            growth_interval=config.training.gradscaler_growth_interval
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    optimizer_cls = torch.optim.AdamW

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length, # 77
        ).input_ids.to(device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sampling.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.training.batch_size, 1, 1)

    unet.to(device)
    unet = DDP(unet, device_ids=[local_rank])
    if not config.model.no_flow:
        res_logflowscore_model.to(device)
        res_logflowscore_model = DDP(res_logflowscore_model, device_ids=[local_rank])

    #######################################################
    #################### FOR GFN ##########################
    if not config.model.no_flow:
        params = [
            {"params": pf_params, "lr": config.training.lr},
            {"params": flow_params, "lr": config.training.flow_lr, 'weight_decay': config.training.flow_wd},
        ]
    else:
        params = [
            {"params": pf_params, "lr": config.training.lr},
        ]

    optimizer = optimizer_cls(
        params,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    return config, pipeline, optimizer, unet, res_logflowscore_model, sample_neg_prompt_embeds, train_neg_prompt_embeds, logger, scaler


def train():

    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config, pipeline, optimizer, unet, res_logflowscore_model, sample_neg_prompt_embeds, train_neg_prompt_embeds, logger, scaler = setup(local_rank, is_local_main_process)

    def decode(latents, clamp=True):
        image = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        image = image / 2.0 + 0.5
        if clamp:
            image = image.clamp(0, 1)
        return image

    device = torch.device(local_rank)

    # prepare prompt and reward fn
    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    reward_fn = getattr(lib.reward_func.rewards, config.experiment.reward_fn)(torch.float32, device)

    def flow_cast_float32():
        return torch.cuda.amp.autocast(dtype=torch.float32)

    autocast = contextlib.nullcontext # LoRA weights are actually float32, but other part of SD are in bf16/fp16
    if config.model.reverse_loss_scale == 0:
        ref_compute_mode = torch.inference_mode
    else:
        ref_compute_mode = contextlib.nullcontext

    result = defaultdict(dict)
    result["config"] = config.to_dict()
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config.sampling.batch_size * num_processes
        * config.sampling.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.training.batch_size * num_processes
        * config.training.gradient_accumulation_steps
    )

    if is_local_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {config.training.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sampling.batch_size}")
        logger.info(f"  Train batch size per device = {config.training.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
        )
        logger.info("")
        logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {config.training.num_inner_epochs}")

    assert config.sampling.batch_size >= config.training.batch_size
    assert config.sampling.batch_size % config.training.batch_size == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = -1 ## epoch -1 only to collect data; training starts from epoch 0
    global_step = 0
    curr_samples = None

    num_inference_steps = config.sampling.num_steps
    scheduler_dt = pipeline.scheduler.timesteps[0] - pipeline.scheduler.timesteps[1]
    num_train_timesteps = int(num_inference_steps * config.model.timestep_fraction) + 1
    accumulation_steps = config.training.gradient_accumulation_steps * num_train_timesteps

    for epoch in range(first_epoch, config.training.num_epochs):

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        unet.zero_grad()
        unet.eval()
        if not config.model.no_flow:
            res_logflowscore_model.zero_grad()

        samples = []
        prompts = []
        with torch.inference_mode():
            for i in tqdm(
                range(config.sampling.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[
                        prompt_fn(**config.experiment.prompt_fn_kwargs)
                        for _ in range(config.sampling.batch_size)
                    ]
                )

                # encode prompts
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                # sample
                with autocast():
                    ret_tuple = sample_trajectory(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=config.sampling.guidance_scale,
                        eta=config.sampling.eta,
                        output_type="pt",
                        return_unetoutput=config.model.unet_reg_scale > 0.,
                    )

                if config.model.unet_reg_scale > 0:
                    images, _, latents, scores, unet_outputs = ret_tuple
                    unet_outputs = torch.stack(unet_outputs, dim=1)  # (batch_size, num_steps, 3, 32, 32)
                else:
                    images, _, latents, scores = ret_tuple

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                scores = torch.stack(scores, dim=1)  # (batch_size, num_steps, 1)
                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sampling.batch_size, 1
                )  # (bs, num_steps)  (981, 961, ..., 21, 1) corresponds to "next_latents"
                step_index = torch.ones(timesteps.size(1), device=timesteps.device, dtype=torch.int64).view(1, -1).expand(timesteps.size(0), -1)

                rewards = reward_fn(images.float(), prompts, prompt_metadata) # (reward, reward_metadata)
                samples.append(
                    {
                        "prompts": prompts, # tuple of strings
                        "prompt_metadata": prompt_metadata,
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[
                            :, :-1
                        ],
                        "next_latents": latents[
                            :, 1:
                        ],
                        "scores": scores,
                        "rewards": rewards,
                        "step_index": step_index
                    }
                )
                if config.model.unet_reg_scale > 0:
                    samples[-1]["unet_outputs"] = unet_outputs


            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"]
                sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:
                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            if epoch >= 0:
                # this is a hack to force wandb to log the images as JPEGs instead of PNGs
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, image in enumerate(images):
                        pil = Image.fromarray(
                            (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((256, 256))
                        pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                    if config.logging.use_wandb and is_local_main_process:
                        wandb.log(
                            {
                                "images": [
                                    wandb.Image(
                                        os.path.join(tmpdir, f"{i}.jpg"),
                                        caption=f"{prompt} | {reward:.2f}",
                                    )
                                    for i, (prompt, reward) in enumerate(
                                        zip(prompts, rewards)
                                    )
                                ],
                            },
                            step=global_step,
                        )

                rewards = torch.zeros(world_size * len(samples["rewards"]),
                            dtype=samples["rewards"].dtype, device=device)
                dist.all_gather_into_tensor(rewards, samples["rewards"])
                rewards = rewards.detach().cpu().float().numpy()
                result["reward_mean"][global_step] = rewards.mean()
                result["reward_std"][global_step] = rewards.std()

                if is_local_main_process:
                    logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                    if config.logging.use_wandb:
                        wandb.log(
                            {
                                "reward_mean": rewards.mean(),
                                "reward_std": rewards.std(),
                            },
                            step=global_step,
                        )

                del samples["prompt_ids"]

                total_batch_size, num_timesteps = samples["timesteps"].shape
                assert (
                    total_batch_size
                    == config.sampling.batch_size * config.sampling.num_batches_per_epoch
                )
                assert num_timesteps == num_inference_steps
            


        ### No sampling for Epoch -1 
        if curr_samples is None:
            curr_samples = samples
            continue

        #################### TRAINING ####################
        for inner_epoch in range(config.training.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    curr_samples[k] = [v[i] for i in perm]
                elif k in ["unet_outputs"]:
                    curr_samples[k] = v[perm]
                else:
                    curr_samples[k] = v[perm]

            if config.model.timestep_fraction < 1:
                if config.sampling.low_var_subsampling:
                    n_trunks = int(num_inference_steps * config.model.timestep_fraction)
                    assert n_trunks >= 1, "Must have at least one trunk"
                    assert num_inference_steps % n_trunks == 0, "num_inference_steps must be divisible by n_trunks"

                    trunk_size = num_inference_steps // n_trunks
                    step_indices = torch.arange(num_inference_steps, device=device)
                    trunks = step_indices.view(n_trunks, trunk_size)  # shape: (n_trunks, trunk_size)

                    # Precompute trunk access pattern (reversed order, repeated)
                    trunk_order = list(reversed(range(n_trunks))) * trunk_size  # len = num_inference_steps

                    perms_list = []
                    for _ in range(total_batch_size):
                        tmp = []
                        for i in trunk_order:
                            trunk = trunks[i]
                            index = torch.randint(0, trunk_size, (1,))
                            tmp.append(trunk[index])
                        interleaved = torch.cat(tmp)
                        perms_list.append(torch.cat([torch.tensor([num_inference_steps - 1], device=device), interleaved]))

                    perms = torch.stack(perms_list)  # shape: (batch_size, 1 + chunk_size * n_trunks)
                else:
                    perms = torch.stack(
                        [
                            torch.randperm(num_timesteps - 1, device=device)
                            for _ in range(total_batch_size)
                        ]
                    ) # (total_batch_size, num_steps)
                    perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms], dim=1)
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                ) # (total_batch_size, num_steps)

            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "next_latents", "scores", "step_index"]
            for key in key_ls:
                curr_samples[key] = curr_samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
            if config.model.unet_reg_scale > 0:
                curr_samples["unet_outputs"] = \
                    curr_samples["unet_outputs"][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config.training.batch_size]
                                for i in range(0, len(v), config.training.batch_size)]
                elif k in ["unet_outputs"]:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])
                else:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            unet.train()
            if not config.model.no_flow:
                res_logflowscore_model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([1, 77, 768])),
                (int64) ('timesteps', torch.Size([1, 50])),
                (bf16) ('latents', torch.Size([1, 50, 4, 64, 64])), ('next_latents', torch.Size([1, 50, 4, 64, 64])),
                ]
                """
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds, sample["prompt_embeds"]]
                )

                buffer = []
                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    with autocast():
                        latent_tmp = sample["latents"][:, j].clone().detach()
                        latent_tmp.requires_grad_(True)

                        # Before inference, disable the LoRA adapters
                        unet.module.disable_adapters()  # This should deactivate any applied LoRA adapter

                        with ref_compute_mode():
                            noise_pred_ref = unet(
                                torch.cat([latent_tmp] * 2),
                                torch.cat([sample["timesteps"][:, j]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                            noise_pred_ref = (
                                    noise_pred_uncond_ref
                                    + config.sampling.guidance_scale
                                    * (noise_pred_text_ref - noise_pred_uncond_ref)
                            )
                            noise_pred_uncond_ref = noise_pred_text_ref = None

                        with torch.inference_mode():
                            _, score_pf_ref = inference_step(
                                pipeline.scheduler, noise_pred_ref,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )

                        if config.model.reverse_loss_scale > 0:
                            _, score_pf_ref_reverse = inference_step(
                                pipeline.scheduler, noise_pred_ref,
                                sample["timesteps"][:, j],
                                latent_tmp,
                                eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                reverse_grad=True,
                                retain_graph=False,
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )
                            score_pf_ref_reverse = score_pf_ref_reverse.detach()
                        _ = noise_pred_ref = None


                        # Optionally, you can re-enable the LoRA adapters after inference if needed
                        unet.module.enable_adapters()  # Re-apply LoRA configuration
                        unet.module.set_adapter("pf")

                        noise_pred = unet(
                            torch.cat([latent_tmp] * 2),
                            torch.cat([sample["timesteps"][:, j]] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = (
                                noise_pred_uncond
                                + config.sampling.guidance_scale
                                * (noise_pred_text - noise_pred_uncond)
                        )
                        noise_pred_uncond = noise_pred_text = None
                        if config.model.unet_reg_scale > 0:
                            unetdiff = (noise_pred - sample["unet_outputs"][:, j]).pow(2)
                            unetreg = torch.mean(unetdiff, dim=(1, 2, 3))
                            unetdiffnorm = unetdiff.sum(dim=[1,2,3]).sqrt()

                        _, score_pf = inference_step(
                            pipeline.scheduler, noise_pred,
                            sample["timesteps"][:, j],
                            latent_tmp,
                            eta=config.sampling.eta,
                            prev_sample=sample["next_latents"][:, j],
                            step_index=sample["step_index"][:, j],
                        )
                        _ = None

                        if config.model.reverse_loss_scale > 0:
                            _, score_pf_reverse = inference_step(
                                pipeline.scheduler, noise_pred,
                                sample["timesteps"][:, j],
                                latent_tmp, eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                reverse_grad=True,
                                retain_graph=True,
                                allow_2nd=False,
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )
                        _ = None

                        #######################################################
                        #################### GFN ALGORITHM ####################
                        #######################################################
                        timestep_next = torch.clamp(sample["timesteps"][:, j] - scheduler_dt, min=0)
                        end_mask = sample["timesteps"][:, j] == pipeline.scheduler.timesteps[-1] # RHS is 1

                        latent_next_tmp = sample["next_latents"][:, j].detach().clone()
                        latent_next_tmp.requires_grad_()
                        unet.module.set_adapter("pf")


                        noise_pred_next_tmp = unet(
                            torch.cat([latent_next_tmp] * 2),
                            torch.cat([timestep_next] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond_next_tmp, noise_pred_next_text_tmp = noise_pred_next_tmp.chunk(2)
                        noise_pred_next_tmp = (
                                noise_pred_uncond_next_tmp
                                + config.sampling.guidance_scale
                                * (noise_pred_next_text_tmp - noise_pred_uncond_next_tmp)
                        )
                        noise_pred_uncond_next_tmp = noise_pred_next_text_tmp = None
                        pred_z0_next = predict_clean(
                            pipeline.scheduler,
                            noise_pred_next_tmp,
                            latent_next_tmp,
                            timestep_next
                        )
                        noise_pred_next_tmp = None
                        pred_xdata_next = decode(pred_z0_next).float()

                        with torch.cuda.amp.autocast(enabled=False):
                            logr_next_tmp = reward_fn(pred_xdata_next, prompts, prompt_metadata)[0]
                            score_r_next_tmp = torch.autograd.grad(
                                outputs=logr_next_tmp.sum(),    # The value whose gradient we want
                                inputs=latent_next_tmp,         # The intermediate node we want the gradient with respect to
                                retain_graph=False,             # Retain graph for further gradient computations
                                create_graph=False              # If higher-order gradients are needed
                            )[0].detach()
                            latent_next_tmp = None
                            score_r_next = config.model.reward_scale * score_r_next_tmp
                            alpha_prod_next = get_alpha_prod_t(pipeline.scheduler, timestep_next, sample["next_latents"][:, j])
                            if config.model.reward_adaptive_mode == 'squared':
                                score_r_next = score_r_next * alpha_prod_next
                            else:
                                score_r_next = score_r_next * alpha_prod_next.sqrt()

                        score_r_next_tmp = None

                        if config.model.reverse_loss_scale > 0:
                            pred_z0 = predict_clean(pipeline.scheduler, noise_pred, latent_tmp, sample["timesteps"][:, j])
                            noise_pred = None
                            pred_xdata = decode(pred_z0).float()
                            with torch.cuda.amp.autocast(enabled=False):
                                logr_tmp = reward_fn(pred_xdata, prompts, prompt_metadata)[0]
                                score_r_tmp = torch.autograd.grad(
                                    outputs=logr_tmp.sum(),    # The value whose gradient we want
                                    inputs=latent_tmp,         # The intermediate node we want the gradient with respect to
                                    retain_graph=True,         # Retain graph for further gradient computations
                                    create_graph=False         # If higher-order gradients are needed
                                )[0].detach()
                                latent_tmp = None
                                score_r = config.model.reward_scale * score_r_tmp
                            score_r_tmp = None
                            alpha_prod = get_alpha_prod_t(pipeline.scheduler, sample["timesteps"][:, j], sample["next_latents"][:, j])
                            if config.model.reward_adaptive_mode == 'squared':
                                score_r = score_r * alpha_prod
                            else:
                                score_r = score_r * alpha_prod.sqrt()

                    if not config.model.no_flow:
                        with flow_cast_float32():
                            if config.experiment.reward_fn == 'aesthetic':
                                flow_prompt_embeds = torch.zeros_like(sample["prompt_embeds"]).float()
                            else:
                                flow_prompt_embeds = sample["prompt_embeds"].float()
                            res_logflow_next = res_logflowscore_model(sample["next_latents"][:, j].float(), timestep_next, flow_prompt_embeds).sample
                            res_logflow = res_logflowscore_model(sample["latents"][:, j].float(), sample["timesteps"][:, j], flow_prompt_embeds).sample

                    if config.model.no_flow:
                        score_pf_target = (score_pf_ref.float() + score_r_next).float()
                    else:
                        score_pf_target = (score_pf_ref.float() + res_logflow_next + score_r_next).float()
                    score_pf_target[end_mask] = (score_pf_ref[end_mask].float() + score_r_next[end_mask].float()).detach()

                    if config.model.reverse_loss_scale > 0:
                        score_pf_reverse_target = (score_pf_ref_reverse.float() - res_logflow - score_r.detach()).float()
                    else:
                        flow_reverse_target = None

                    if config.model.reverse_loss_scale > 0:
                        score_pf_reverse_target[end_mask] = torch.zeros_like(score_pf_reverse_target[end_mask])

                    with torch.inference_mode():
                        grad_norm_score_ref = score_pf_ref.pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_res_score = (score_pf - score_pf_ref).pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_r = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()
                        if not config.model.no_flow:
                            grad_norm_resflow = res_logflow_next.pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_pf_target = score_pf_target.pow(2).sum(dim=[1,2,3]).sqrt()
                        if config.model.reverse_loss_scale > 0:
                            grad_norm_score_pf_reverse_target = score_pf_reverse_target.pow(2).sum(dim=[1,2,3]).sqrt()

                    score_pf_ref = None
                    score_r_next = None
                    score_r = None
                    score_pf_ref_reverse = None
                    score_pb_reverse = None


                    if not config.model.no_flow:
                        loss_terminal = (res_logflow_next.pow(2).mean(dim=[1,2,3]) * end_mask.float()).sum() / (end_mask.float().sum() + 1e-6)
                    else:
                        loss_terminal = torch.zeros(1, device=score_pf.device)

                    losses_forward = (score_pf - score_pf_target).pow(2)
                    score_pf = score_pf_target = None
                    loss_forward_mean = losses_forward.mean()

                    if config.model.reverse_loss_scale > 0:
                        losses_backward = (score_pf_reverse - score_pf_reverse_target).pow(2)  # (bs,)
                        score_pf_reverse = score_pf_reverse_target = None
                        loss_backward_mean = losses_backward.mean()
                    else:
                        losses_backward = torch.zeros_like(losses_forward)
                        loss_backward_mean = 0.0

                    losses = (losses_forward + config.model.reverse_loss_scale * losses_backward + loss_terminal).mean()

                    if config.model.unet_reg_scale > 0:
                        losses = losses + config.model.unet_reg_scale * unetreg.mean()
                    loss = torch.mean(losses)

                    loss = loss / accumulation_steps
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()


                    #### Log
                    info["loss"].append(loss_forward_mean + loss_backward_mean)
                    info["loss_terminal"].append(loss_terminal)
                    info["loss_forward"].append(loss_forward_mean.detach())
                    if config.model.reverse_loss_scale > 0:
                        info["losses_backward"].append(loss_backward_mean.detach())

                    with torch.inference_mode():
                        info["norm_score_ref_mean"].append(grad_norm_score_ref.mean())
                        info["norm_score_ref_min"].append(grad_norm_score_ref.min())
                        info["norm_score_ref_max"].append(grad_norm_score_ref.max())
                        info["norm_score_residual_mean"].append(grad_norm_res_score.mean())
                        info["norm_score_residual_min"].append(grad_norm_res_score.min())
                        info["norm_score_residual_max"].append(grad_norm_res_score.max())
                        info["norm_score_r_mean"].append(grad_norm_score_r.mean())
                        info["norm_score_r_min"].append(grad_norm_score_r.min())
                        info["norm_score_r_max"].append(grad_norm_score_r.max())
                        if config.model.unet_reg_scale > 0:
                            info["norm_unet_diff_mean"].append(unetdiffnorm.mean())
                            info["norm_unet_diff_mean"].append(unetdiffnorm.min())
                            info["norm_unet_diff_mean"].append(unetdiffnorm.max())

                    info["losses_forward_max"].append(losses_forward.max())
                    info["losses_backward_max"].append(losses_backward.max())
                    info["losses_bidir_max"].append((losses_forward + config.model.reverse_loss_scale * losses_backward).max())
                    if config.model.unet_reg_scale > 0:
                        info["unetreg"].append(unetreg.mean().detach())


                    # prevent OOM
                    image = None
                    noise_pred_uncond = noise_pred_text = noise_pred = None
                    logr_next_tmp = logr_tmp = None
                    _ = log_pf = log_pb = None
                    score_pb_reverse = None
                    unetreg = losses =  None
                    score_pf_ref = score_pf = None
                    score_r_next = score_r_next_tmp = None
                    noise_pred_uncond_ref = noise_pred_text_ref = noise_pred_ref = None
                    score_pf_target = None
                    res_logflow = res_logflow_next = None
                    grad_norm_score_pf_target = grad_norm_score_pf_reverse_target = None



                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config.training.gradient_accumulation_steps == 0):
                    if scaler:
                        scaler.unscale_(optimizer)
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in unet.named_parameters() if '.pf.' in name], config.training.max_grad_norm)
                        if not config.model.no_flow:
                            flow_update_grad = torch.nn.utils.clip_grad_norm_(res_logflowscore_model.parameters(), config.training.max_grad_norm)

                        scaler.step(optimizer)
                        optimizer.step()
                        scaler.update()
                    else:
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in unet.named_parameters() if '.pf.' in name], config.training.max_grad_norm)
                        if not config.model.no_flow:
                            flow_update_grad = torch.nn.utils.clip_grad_norm_(res_logflowscore_model.parameters(), config.training.max_grad_norm)

                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    ### avoid memory leak
                    if not config.model.no_flow:
                        for param in res_logflowscore_model.parameters():
                            param.grad = None
                    for param in unet.parameters():
                        param.grad = None

                    old_info = info
                    info = {}
                    for k, v in old_info.items():
                        if '_min' in k:
                            info[k] = torch.min(torch.stack(v))
                        elif '_max' in k:
                            info[k] = torch.max(torch.stack(v))
                        else:
                            try:
                                info[k] = torch.mean(torch.stack(v))
                            except Exception as e:
                                print(k)
                                print(v)
                                raise e

                    dist.barrier()
                    for k, v in info.items():
                        if '_min' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MIN)
                        elif '_max' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                        else:
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {k: v / num_processes if ('_min' not in k and '_max' not in k) else v for k, v in info.items()}
                    for k, v in info.items():
                        result[k][global_step] = v.item()

                    info.update({"epoch": epoch})
                    info.update({"global_step": global_step})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time


                    if is_local_main_process:
                        if scaler:
                            info.update({"grad_scale": scaler.get_scale()})
                            result["grad_scale"] = scaler.get_scale()


                    if is_local_main_process:
                        if config.logging.use_wandb:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.6f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict


        curr_samples = samples
        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(config.saving.output_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config.logging.save_freq == 0 or epoch == config.training.num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch{epoch}")
                unwrapped_unet = unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                conv_out_weights = unwrapped_unet.conv_out.state_dict()  # Extract only the weights of the conv_out layer
                torch.save(conv_out_weights, os.path.join(config.saving.output_dir, f"conv_out_weights_epoch{epoch}.pt"))
                logger.info(f"Saved state to {save_path}")

            dist.barrier()

    if is_local_main_process:
        save_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch{epoch}")
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            is_main_process=is_local_main_process,
            safe_serialization=True,
        )
        conv_out_weights = unwrapped_unet.conv_out.state_dict()  # Extract only the weights of the conv_out layer
        torch.save(conv_out_weights, os.path.join(config.saving.output_dir, f"conv_out_weights_epoch{epoch}.pt"))
        logger.info(f"Saved state to {save_path}")
    dist.barrier()

    if config.logging.use_wandb and is_local_main_process:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)