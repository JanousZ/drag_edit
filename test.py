#!/usr/bin/env python3
"""
Training script for Kontext model with DeepSpeed and Accelerate
"""

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
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
from diffusers import FluxPipeline
from pipeline_dreamomni2 import DreamOmni2Pipeline
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
from argparse import ArgumentParser
from module.point import PointsMapEncoder
from module.dit import FluxTransformer2DPointsModel
import json
import numpy as np
from PIL import Image
from accelerate.utils import set_module_tensor_to_device

def main():
    parser = ArgumentParser()
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--root_dir", type=str, default="./output")
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    base_model = "/mnt/disk1/models/FLUX.1-Kontext-dev"
    dit = FluxTransformer2DPointsModel.from_pretrained(base_model, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = DreamOmni2Pipeline.from_pretrained(base_model, transformer=dit)
    if not args.no_lora:
        edit_lora_path = "/home/yanzhang/drag_edit/lora_ckpt/checkpoint-22500/lora.safetensors"
        # edit_lora_path = "/home/yanzhang/models/DreamOmni2/edit_lora/pytorch_lora_weights.safetensors"
        edit_lora_dict = load_file(edit_lora_path)
        edit_lora_dict_2 = {k[12:] : v.to(torch.bfloat16) for k,v in edit_lora_dict.items()}  # 去除 transformer.
        for k,v in edit_lora_dict_2.items():
            if "points" in k:
                edit_lora_dict_2[k] = edit_lora_dict_2[k] * 0.01

        lora_rank = 32
        lora_alpha = 32
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v", "to_k", "to_out.0"], lora_dropout=0.05
        )
        dit.add_adapter(lora_config, adapter_name="edit")

        for name,param in dit.named_parameters():
            if param.device.type == "meta":
                safe_data = torch.zeros(param.shape, dtype=param.dtype, device="cuda")
                set_module_tensor_to_device(dit, name, device="cuda", value=safe_data)

        missing, unexpected = dit.load_state_dict(edit_lora_dict_2, strict=False)
    dit.eval()
    pipe.enable_model_cpu_offload()
    # pipe = pipe.to("cuda")
    
    points_map_encoder_config_path = "/home/yanzhang/drag_edit/module/pme_config.json"
    with open(points_map_encoder_config_path, "r") as f:
        points_map_encoder_config = json.load(f)
    points_map_encoder = PointsMapEncoder(**points_map_encoder_config)
    encoder_path = "/home/yanzhang/drag_edit/lora_ckpt/checkpoint-22500/points_map_encoder.safetensors"
    encoder_state_dict = load_file(encoder_path)
    encoder_state_dict = {k: v.to(torch.bfloat16) for k, v in encoder_state_dict.items()}
    missing, unexpected = points_map_encoder.load_state_dict(encoder_state_dict)
    points_map_encoder.eval()
    points_map_encoder = points_map_encoder.to("cuda")

    # 加载数据 
    # dataset = Replace5kDataset(json_file="/home/yanzhang/datasets/replace-5k/test.json", is_image_preprocess=False)
    dataset = DragDataset(jsonl_file="/home/yanzhang/dragdatasets/paired_frames.jsonl")
    
    for i in range(len(dataset)):
        output_dir = os.path.join(args.root_dir, f"{i}")
        os.makedirs(output_dir, exist_ok=True)
        data = dataset[i]
        # src_image = data["src_image"]
        # ref_image = data["ref_image"]
        # data["src_image"].save("./src.jpg")
        # data["ref_image"].save("./ref.jpg")
        img_path1, tgt_image_path = dataset.data[i]["pair"]
        frame1 = os.path.basename(img_path1).split('_frame_')[1].split('.png')[0]
        src_image_path = os.path.join(dataset.data[i]["folder"], f"original_frame_{frame1}.png")
        src_image = Image.open(src_image_path)
        tgt_image = Image.open(tgt_image_path)
        src_image.save(os.path.join(output_dir, "src.jpg"))
        tgt_image.save(os.path.join(output_dir, "gt.jpg"))

        with open(os.path.join(output_dir, "instruction.txt"), "w") as f:
            f.write("Drag the image according to the given points mapping embeddings.")

        # get points_emb
        src_points = torch.from_numpy(data["src_points"]).unsqueeze(0)
        tgt_points = torch.from_numpy(data["tgt_points"]).unsqueeze(0)
        points=torch.concat([tgt_points, src_points], dim=0)
        W, H = src_image.size
        B = 1
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)  #[B, N, 2] for (x,y)
        N = points.shape[1]

        points_map = torch.zeros((2 * B, 1, H, W))
        coords = points.long()
        xs = coords[..., 0].clamp(0, W - 1)  #[2B, N]
        ys = coords[..., 1].clamp(0, H - 1)  #[2B, N]
        batch_indices = torch.arange(2 * B).view(2 * B, 1).expand(2 * B, N)
        values = torch.arange(1, N + 1).float().expand(2 * B, N)
        points_map[batch_indices, 0, ys, xs] = values
        
        ts_points_emb = points_map_encoder(points_map.to(points_map_encoder.device))
        tgt_points_emb, src_points_emb = ts_points_emb.chunk(2, dim=0) 
        points_emb = torch.concat([tgt_points_emb, src_points_emb], dim=1)

        image = pipe(
            images=[src_image],
            height=src_image.height,
            width=src_image.width,
            prompt=["Drag the image according to the given points mapping embeddings."],
            num_inference_steps=20,
            guidance_scale=3.5,
            max_area=512**2,
            points_emb=points_emb.to(torch.bfloat16),
            _auto_resize=False,
        ).images[0]
        save_name = os.path.join(output_dir, "tgt.jpg" if not args.no_lora else "tgt_no_lora.jpg")
        image.save(save_name)

        break
    

if __name__ == "__main__":
    main()