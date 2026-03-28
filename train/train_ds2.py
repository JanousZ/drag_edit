#!/usr/bin/env python3
"""
Training script for Kontext model with DeepSpeed and Accelerate
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from typing import Dict, Any, Optional
import argparse
from my_datasets.dragdataset import DragDataset, dd_collate_fn
from torch.utils.data import DataLoader
import os
import logging
from accelerate.logging import get_logger
import diffusers
import datasets
import transformers
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from module.dit import FluxTransformer2DPointsModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
    is_wandb_available
)
from safetensors.torch import save_file, load_file
import shutil
from utils.infer_utils import _encode_prompt_with_clip, _encode_prompt_with_t5
import os
import math
from module.point import PointsMapEncoder
import json
import numpy as np

class JointModel(nn.Module):
    def __init__(self, dit, points_map_encoder):
        super().__init__()
        self.dit = dit
        self.points_map_encoder = points_map_encoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,

        mode: str = None,
        img_tensor: torch.Tensor = None, # 指的是直接图片导入未经VAE处理的tensor
        points: torch.Tensor = None,
        weight_dtype = None,
    ):
        B, _, H, W = img_tensor.shape
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)  #[B, N, 2] for (x,y)
        N = points.shape[1]

        if mode == "integer_index":
            points_map = torch.zeros((2 * B, 1, H, W), device=img_tensor.device, dtype=weight_dtype)
            coords = points.long()
            xs = coords[..., 0].clamp(0, W - 1)  #[2B, N]
            ys = coords[..., 1].clamp(0, H - 1)  #[2B, N]
            batch_indices = torch.arange(2 * B, device=img_tensor.device).view(2 * B, 1).expand(2 * B, N)
            values = torch.arange(1, N + 1, device=img_tensor.device).float().expand(2 * B, N).to(dtype=weight_dtype)
            points_map[batch_indices, 0, ys, xs] = values
        
        ts_points_emb = self.points_map_encoder(points_map.to(self.points_map_encoder.device))
        tgt_points_emb, src_points_emb = ts_points_emb.chunk(2, dim=0) 
        points_emb = torch.concat([tgt_points_emb, src_points_emb], dim=1)

        noise_pred = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,     #[0,1]
            guidance=guidance,     #[1.0]
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            points_emb=points_emb.to(dtype=weight_dtype),
            return_dict=False,
        )
        
        return noise_pred

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
    parser.add_argument("--checkpoints_total_limit", type=int, default=20)
    parser.add_argument("--resume_lora_path", type=str, default=None)
    parser.add_argument("--resume_points_map_encoder_path", type=str, default=None)
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

# from PIL import Image,ImageDraw
# def check_match(images, points, save_path="check_points.png"):
#     # images: [b, c, h, w], 假设范围在 [-1, 1]
#     # points: [b, n, 2], 假设坐标是像素坐标 (x, y)
    
#     # 1. 提取第一张图并转换格式
#     img_tensor = images[0].detach().cpu()
#     img_np = (img_tensor + 1) * 127.5
#     img_np = img_np.clamp(0, 255).permute(1, 2, 0).numpy().astype('uint8')
    
#     # 2. 转换为 PIL Image
#     img_pil = Image.fromarray(img_np)
#     draw = ImageDraw.Draw(img_pil)
    
#     # 3. 提取点位坐标
#     pts = points[0].detach().cpu().numpy() # [n, 2]
    
#     # 4. 遍历并画圆
#     radius = 3
#     for x, y in pts:
#         # 定义圆的左上角和右下角坐标
#         left_up = (x - radius, y - radius)
#         right_down = (x + radius, y + radius)
#         # outline 为圆周颜色，fill 为填充颜色
#         draw.ellipse([left_up, right_down], outline="red", width=2)
    
#     # 5. 保存图像
#     img_pil.save(save_path)
#     print(f"Saved visualization to {save_path}")
    

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
    accelerator.init_trackers("drag_edit", config=vars(ARGS))

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

    dit = FluxTransformer2DPointsModel.from_pretrained(base_model, subfolder="transformer")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    t5 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2")
    clip = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    t5_tokenizer = T5Tokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
    clip_tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    points_map_encoder_config_path = "/home/yanzhang/drag_edit/module/pme_config.json"
    with open(points_map_encoder_config_path, "r") as f:
        points_map_encoder_config = json.load(f)
    points_map_encoder = PointsMapEncoder(**points_map_encoder_config)

    t5 = t5.to(accelerator.device)
    clip = clip.to(accelerator.device)
    vae = vae.to(accelerator.device)
    dit._initialize_custom_layers()

    points_map_encoder._initialize_layers()

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    vae.eval()
    t5.eval()
    clip.eval()

    model = JointModel(dit, points_map_encoder)
    model.train()
    model.dit.requires_grad_(False)
    model.points_map_encoder.requires_grad_(True)
    model.dit.enable_gradient_checkpointing()
    
    # 注入 LoRA 层, 先开lora，后开其他层
    lora_rank = ARGS.lora_rank
    lora_alpha = ARGS.lora_alpha
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v", "to_k", "to_out.0"], lora_dropout=0.05
    )
    model.dit.add_adapter(lora_config, adapter_name="edit")
    model.dit.points_embedder.requires_grad_(True)
    model.dit.points_embedder_scale.requires_grad_(True)

    # 断点重训
    if ARGS.resume_lora_path is not None:
        resume_lora_path = ARGS.resume_lora_path
        lora_weights = load_file(resume_lora_path)
        missing_keys, unexpected_keys = model.dit.load_state_dict(lora_weights, strict=False)
        logger.info(f"unexpected_lora_keys:{unexpected_keys}")
        logger.info("successfully resume lora")
    
    if ARGS.resume_points_map_encoder_path is not None:
        resume_path = ARGS.resume_points_map_encoder_path
        weights = load_file(resume_path)
        missing_keys, unexpected_keys = model.points_map_encoder.load_state_dict(weights, strict=True)
        logger.info(f"missing_keys:{missing_keys}")
        logger.info(f"unexpected_lora_keys:{unexpected_keys}")
        logger.info("successfully resume pemb")

    # 检查可训练参数
    params_groups = [
        {"params": [p for p in model.dit.parameters() if p.requires_grad], "lr": ARGS.lr},
        {"params": [p for p in model.points_map_encoder.parameters() if p.requires_grad], "lr": ARGS.lr}, # 插件通常用稍小的 LR
    ]
    optimizer = AdamW(params_groups)

    # 加载数据 
    dataset = DragDataset(jsonl_file="/home/yanzhang/dragdatasets/paired_frames.jsonl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dd_collate_fn)

    # Prepare with accelerator
    accelerator.wait_for_everyone()
    model, optimizer, dataloader= accelerator.prepare(model, optimizer, dataloader)

    # unwrapped_model = accelerator.unwrap_model(model)
    # if ARGS.resume_lora_path:
    # # 注意：这里加载时要确保 device 一致
    #     lora_weights = load_file(ARGS.resume_lora_path)
    #     missing_keys, unexpected_keys = unwrapped_model.dit.load_state_dict(lora_weights, strict=False)
    #     logger.info(f"unexpected_lora_keys:{unexpected_keys}")
    #     logger.info("successfully resume lora")

    # if ARGS.resume_points_map_encoder_path:
    #     weights = load_file(ARGS.resume_points_map_encoder_path)
    #     missing_keys, unexpected_keys = unwrapped_model.points_map_encoder.load_state_dict(weights, strict=True)
    #     logger.info(f"missing_keys:{missing_keys}")
    #     logger.info(f"unexpected_lora_keys:{unexpected_keys}")
    #     logger.info("successfully resume pemb")

    trainable_named_params = [n for n, p in model.named_parameters() if p.requires_grad]
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
                
                src_images = batch["input_image"].to(accelerator.device)
                tgt_images = batch["target_image"].to(accelerator.device)

                src_points = batch["src_points"]
                tgt_points = batch["tgt_points"]

                batch_size = src_images.shape[0]
                instructions = [""] * batch_size  # 或者给一个合适的instrcution

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
                    num_channels_latents = dit.config.in_channels // 4
                    src_image_latents = (vae.encode(src_images.float()).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    tgt_image_latents = (vae.encode(tgt_images.float()).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                    image_latent_height, image_latent_width = src_image_latents.shape[2:]
                    src_image_latents = _pack_latents(
                        src_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                    tgt_image_latents = _pack_latents(
                        tgt_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                    src_image_ids = _prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, image_latent_width // 2, accelerator.device, prompt_embeds.dtype
                    )
                    tgt_image_ids = _prepare_latent_image_ids(
                        batch_size, image_latent_height // 2, image_latent_width // 2, accelerator.device, prompt_embeds.dtype
                    )
                    w_offset = image_latent_width // 2
                    src_image_ids[..., 0] = 1
                    # src_image_ids[..., 2] += w_offset
                    
                    # timestep
                    t_raw = torch.rand(batch_size, 1, 1, device=accelerator.device)   # 可能要加time shift
                    t = time_shift(mu=1.1, sigma=0, t=t_raw)

                    # sample & add_noise
                    x_1 = torch.randn_like(tgt_image_latents).to(accelerator.device)
                    x_t = (1 - t) * tgt_image_latents + t * x_1

                # denoise
                latent_model_input = torch.cat([x_t, src_image_latents], dim=1)
                latent_ids = torch.cat([tgt_image_ids, src_image_ids], dim=0)
                guidance = torch.full((x_t.shape[0],), 1, device=x_t.device)

                # 
                # check_match(src_images, src_points, "src.jpg")
                # check_match(tgt_images, tgt_points, "tgt.jpg")

                # 还需要修改dit的第一个conv_in
                noise_pred = model(
                    hidden_states=latent_model_input.to(dtype=weight_dtype),
                    timestep=t.squeeze(1).squeeze(1).to(dtype=weight_dtype),     #[0,1]
                    guidance=guidance.to(dtype=weight_dtype),     #[1.0]
                    pooled_projections=pooled_prompt_embeds.to(dtype=weight_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    txt_ids=text_ids.to(dtype=weight_dtype),
                    img_ids=latent_ids.to(dtype=weight_dtype),
                    return_dict=False,

                    mode="integer_index",
                    img_tensor=src_images, # 指的是直接图片导入未经VAE处理的tensor
                    points=torch.concat([tgt_points, src_points], dim=0).to(dtype=weight_dtype),
                    weight_dtype=weight_dtype,
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
                    accelerator.log({"scale": model.dit.points_embedder_scale.detach().float().cpu().item()})
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
                        lora_state_dict = {"transformer." + k: unwrapped_model_state[k] for k in unwrapped_model_state.keys() if 'lora' in k or 'points' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

                        unwrapped_model_state = accelerator.unwrap_model(points_map_encoder).state_dict()
                        save_file(
                            unwrapped_model_state,
                            os.path.join(save_path, "points_map_encoder.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

            if global_step >= ARGS.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()