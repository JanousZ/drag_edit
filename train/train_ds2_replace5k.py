#!/usr/bin/env python3
"""
Training script for Kontext model with DeepSpeed and Accelerate
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
import argparse
from my_datasets.replace5k import Replace5kDataset
from torch.utils.data import DataLoader
import logging
from accelerate.logging import get_logger
import diffusers
import datasets
import transformers
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
    is_wandb_available
)
from safetensors.torch import save_file, load_file
import shutil
from utils.infer_utils import _encode_prompt_with_clip, _encode_prompt_with_t5
import math

os.environ["WANDB_API_KEY"] = "86ab58d2a525a27f7a60ab5fa492d36bdf932255"

def parse_args():
    """Parses command-line arguments for model paths and server configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="/mnt/disk1/models/FLUX.1-Kontext-dev", 
        help="Path to the FLUX.1-Kontext editing."
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./lora_ckpt")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--resume_lora_path", type=str, default=None)
    args = parser.parse_args()
    return args

# Helper functions
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.to(device=device, dtype=dtype)

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    # Flux 推荐的 shift 逻辑
    return math.exp(mu) * t / (1 + (math.exp(mu) - 1) * t)

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

def main():
    ARGS = parse_args()
    logging_dir = os.path.join(ARGS.output_dir, ARGS.logging_dir)

    # 初始化分布式环境
    accelerator = Accelerator(
        log_with=ARGS.report_to if ARGS.report_to != "none" else None,
        project_dir=ARGS.output_dir,
    )
    torch.cuda.set_device(accelerator.device)

    accelerator.init_trackers("ds2_kontext", config=vars(ARGS))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if ARGS.output_dir is not None:
            os.makedirs(ARGS.output_dir, exist_ok=True)

    # 加载预训练模型
    base_model = ARGS.base_model_path

    dit = FluxTransformer2DModel.from_pretrained(base_model, subfolder="transformer")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    t5 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2")
    clip = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    t5_tokenizer = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2")
    clip_tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")

    t5 = t5.to(accelerator.device)
    clip = clip.to(accelerator.device)
    vae = vae.to(accelerator.device)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)

    vae.eval()
    t5.eval()
    clip.eval()
    dit.train()
    dit.enable_gradient_checkpointing()

    # 注入 LoRA 层
    lora_rank = ARGS.lora_rank
    lora_alpha = ARGS.lora_alpha
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v", "to_k", "to_out.0"], lora_dropout=0.05
    )
    dit.add_adapter(lora_config, adapter_name="edit")

    # 断点重训
    if ARGS.resume_lora_path is not None:
        resume_lora_path = ARGS.resume_lora_path
        lora_weights = load_file(resume_lora_path)
        missing_keys, unexpected_keys = dit.load_state_dict(lora_weights, strict=False)
        logger.info(f"unexpected_lora_keys:{unexpected_keys}")
        logger.info("successfully resume lora")

    # 检查可训练参数
    trainable_params = [p for n, p in dit.named_parameters() if p.requires_grad]

    # 设置优化器
    lr = ARGS.lr
    optimizer = AdamW(trainable_params, lr=lr)

    # 加载数据 
    dataset = Replace5kDataset(json_file="/home/yanzhang/datasets/replace-5k/train.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    # Prepare with accelerator
    accelerator.wait_for_everyone()
    dit, optimizer, dataloader= accelerator.prepare(dit, optimizer, dataloader)
    dit.train()

    trainable_named_params = [n for n, p in dit.named_parameters() if p.requires_grad]
    if accelerator.is_main_process:
        logger.info("Here are the trainable params")
        logger.info(trainable_named_params)

    # 训练循环
    logger.info("***** Running training *****")
    num_epochs = ARGS.num_epochs
    global_step = 0
    save_steps = ARGS.save_steps

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(dit):
                instructions = batch["prompt"]
                src_images = batch["src_image"].to(accelerator.device)
                ref_images = batch["ref_image"].to(accelerator.device)
                tgt_images = batch["tgt_image"].to(accelerator.device)
                batch_size = src_images.shape[0]

                with torch.no_grad():
                    prompts = instructions
                    
                    # 获取文本embedding & tokens
                    pooled_prompt_embeds = _encode_prompt_with_clip(
                        text_encoder=clip,
                        tokenizer=clip_tokenizer,
                        prompt=prompts,
                        device=accelerator.device,
                    )

                    prompt_embeds = _encode_prompt_with_t5(
                        text_encoder=t5,
                        tokenizer=t5_tokenizer,
                        prompt=prompts,
                        device=accelerator.device,
                    )

                    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=clip.dtype)
                    
                    # image encode
                    num_channels_latents = dit.module.config.in_channels // 4
                    src_image_latents = (vae.encode(src_images.float()).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    ref_image_latents = (vae.encode(ref_images.float()).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    tgt_image_latents = (vae.encode(tgt_images.float()).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    image_latent_height, image_latent_width = src_image_latents.shape[2:]
                    src_image_latents = _pack_latents(
                        src_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                    ref_image_latents = _pack_latents(
                        ref_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                    tgt_image_latents = _pack_latents(
                        tgt_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                    src_image_ids = _prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, image_latent_width // 2, accelerator.device, prompt_embeds.dtype
                    )
                    ref_image_ids = _prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, image_latent_width // 2, accelerator.device, prompt_embeds.dtype
                    )
                    tgt_image_ids = _prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, image_latent_width // 2, accelerator.device, prompt_embeds.dtype
                    )
                    w_offset = image_latent_width // 2
                    src_image_ids[..., 0] = 1
                    # src_image_ids[..., 2] += w_offset
                    w_offset += image_latent_width // 2
                    ref_image_ids[..., 0] = 2
                    # ref_image_ids[..., 2] += w_offset
                    
                    # timestep
                    t_raw = torch.rand(batch_size, 1, 1, device=accelerator.device)   # 可能要加time shift
                    t = time_shift(mu=3, sigma=0, t=t_raw)

                    # sample & add_noise
                    x_1 = torch.randn_like(tgt_image_latents).to(accelerator.device)
                    x_t = (1 - t) * tgt_image_latents + t * x_1

                # denoise
                latent_model_input = torch.cat([x_t, src_image_latents, ref_image_latents], dim=1)
                latent_ids = torch.cat([tgt_image_ids, src_image_ids, ref_image_ids], dim=0)
                guidance = torch.full((x_t.shape[0],), 1, device=x_t.device)

                noise_pred = dit(
                    hidden_states=latent_model_input.to(dtype=weight_dtype),
                    timestep=t.squeeze(1).squeeze(1).to(dtype=weight_dtype),     #[0,1]
                    guidance=guidance.to(dtype=weight_dtype),     #[1.0]
                    pooled_projections=pooled_prompt_embeds.to(dtype=weight_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    txt_ids=text_ids.to(dtype=weight_dtype),
                    img_ids=latent_ids.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :x_t.size(1)]

                # loss
                diff_loss = torch.nn.functional.mse_loss(noise_pred.float(), (x_1 - tgt_image_latents).float(), reduction="mean")
                loss = diff_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                if ARGS.report_to != "none":
                    accelerator.log({"loss": loss.item()}, step=global_step)
                logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")

                if global_step % save_steps == 0:
                    save_path = os.path.join(ARGS.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    # accelerator.save_state(save_path)

                    if accelerator.is_main_process:
                        if ARGS.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(ARGS.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= ARGS.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - ARGS.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(ARGS.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()
                        lora_state_dict = {"transformer." + k: unwrapped_model_state[k] for k in unwrapped_model_state.keys() if 'lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

            if global_step >= ARGS.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()