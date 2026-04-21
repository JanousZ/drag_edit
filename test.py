import torch
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
import argparse
from my_datasets.dragdataset import DragDataset, dd_collate_fn, DragBenchDataset
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
import cv2

def visualize_drag_points(src_img_pil, tgt_img_pil, src_pts, tgt_pts, save_path):
    """
    将 src 和 tgt 点标注在图像上并水平拼接保存
    src_pts/tgt_pts: [N, 2] 形状的 tensor 或 array
    """
    def pil_to_cv2(img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    s_img = pil_to_cv2(src_img_pil)
    t_img = pil_to_cv2(tgt_img_pil)
    
    # 确保坐标是整数
    s_pts = src_pts.cpu().numpy().astype(int)
    t_pts = tgt_pts.cpu().numpy().astype(int)
    
    # 颜色配置: Src用红色(Red), Tgt用绿色(Green)
    # OpenCV 颜色空间是 BGR
    for i in range(len(s_pts)):
        # 在原图画起始点 (Red)
        cv2.circle(s_img, (s_pts[i, 0], s_pts[i, 1]), 4, (0, 0, 255), -1)
        # 在目标图画终点 (Green)
        cv2.circle(t_img, (t_pts[i, 0], t_pts[i, 1]), 4, (0, 255, 0), -1)
        # 画个小箭头显示趋势
        cv2.arrowedLine(s_img, (s_pts[i, 0], s_pts[i, 1]), 
                       (t_pts[i, 0], t_pts[i, 1]), (0, 255, 255), 1, tipLength=0.2)

    # 水平拼接显示
    combined = np.hstack((s_img, t_img))
    cv2.imwrite(save_path, combined)

def augment_drag_points(src_points, tgt_points, radius, num_neighbors, img_w, img_h):
    """
    对每个原始拖拽点，在其附近采样邻居点，施加相同位移向量，扩充点对数量。
    src_points: [N, 2] numpy array (x, y)
    tgt_points: [N, 2] numpy array (x, y)
    返回增强后的 src_points, tgt_points: [N*(1+num_neighbors), 2]
    """
    aug_src = [src_points]
    aug_tgt = [tgt_points]
    for i in range(len(src_points)):
        displacement = tgt_points[i] - src_points[i]  # 位移向量
        for _ in range(num_neighbors):
            # 在原始src点附近随机采样偏移
            offset = np.random.randint(-radius, radius + 1, size=2)
            new_src = src_points[i] + offset
            new_tgt = new_src + displacement  # 同向迁移
            # 确保点在图像范围内
            new_src[0] = np.clip(new_src[0], 0, img_w - 1)
            new_src[1] = np.clip(new_src[1], 0, img_h - 1)
            new_tgt[0] = np.clip(new_tgt[0], 0, img_w - 1)
            new_tgt[1] = np.clip(new_tgt[1], 0, img_h - 1)
            aug_src.append(new_src.reshape(1, 2))
            aug_tgt.append(new_tgt.reshape(1, 2))
    return np.concatenate(aug_src, axis=0), np.concatenate(aug_tgt, axis=0)


def tensor_to_Image(x):
    x = (x + 1.0) * 127.5
    x = x.permute(1, 2, 0)
    x = torch.clamp(x, 0, 255).to(torch.uint8)
    x = Image.fromarray(x.numpy())
    return x


def main():
    parser = ArgumentParser()
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--lora_file", type=str, default="lora.safetensors")
    parser.add_argument("--encoder_file", type=str, default="points_map_encoder.safetensors")
    parser.add_argument("--encoder_config", type=str, default="./module/pme_config.json")
    parser.add_argument("--dataset_jsonl", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="/mnt/disk1/models/FLUX.1-Kontext-dev")
    parser.add_argument("--dataset_type", type=str, default="drag", choices=["drag", "dragbench"],
                        help="Dataset type: 'drag' for DragDataset, 'dragbench' for DragBenchDataset")
    parser.add_argument("--dragbench_root", type=str, default="/mnt/disk1/datasets/DragBench")
    parser.add_argument("--annotation_variant", type=str, default="meta_data.pkl",
                        help="meta file name inside each sample dir; use e.g. meta_data_multi.pkl to switch to manual multi-point annotation")
    parser.add_argument("--only_annotated", action="store_true",
                        help="only iterate over samples that have the specified annotation_variant file")
    parser.add_argument("--reverse_direction", action="store_true",
                        help="(drag dataset only) swap src<->tgt image & points to evaluate backward drag prediction")
    parser.add_argument("--augment_points", action="store_true", help="Enable nearby point augmentation")
    parser.add_argument("--augment_radius", type=int, default=5, help="Radius to sample nearby points")
    parser.add_argument("--augment_num", type=int, default=2, help="Number of nearby points per original point")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_model = args.base_model
    dit = FluxTransformer2DPointsModel.from_pretrained(base_model, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = DreamOmni2Pipeline.from_pretrained(base_model, transformer=dit)
    if args.use_lora:
        edit_lora_path = os.path.join(args.checkpoint_dir, args.lora_file)
        # edit_lora_path = "/home/yanzhang/models/DreamOmni2/edit_lora/pytorch_lora_weights.safetensors"
        edit_lora_dict = load_file(edit_lora_path)
        edit_lora_dict_2 = {k[12:] : v.to(torch.bfloat16) for k,v in edit_lora_dict.items()}  # 去除 transformer.
        for k,v in edit_lora_dict_2.items():
            if "points" in k:
                edit_lora_dict_2[k] = edit_lora_dict_2[k]

        lora_rank = 32
        lora_alpha = 32
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v", "to_k", "to_out.0"], lora_dropout=0.05
        )
        dit.add_adapter(lora_config, adapter_name="edit")

        for name,param in dit.named_parameters():
            if param.is_meta:
                safe_data = torch.zeros(param.shape, dtype=param.dtype, device="cuda")
                set_module_tensor_to_device(dit, name, device="cuda", value=safe_data)

        missing, unexpected = dit.load_state_dict(edit_lora_dict_2, strict=False)
        # print(f"Missing keys when loading LoRA weights: {missing}")
        print(f"Unexpected keys when loading LoRA weights: {unexpected}")
    else:
        pass

    dit.eval()
    pipe.enable_model_cpu_offload()
    # pipe = pipe.to("cuda")
    
    with open(args.encoder_config, "r") as f:
        points_map_encoder_config = json.load(f)
    points_map_encoder = PointsMapEncoder(**points_map_encoder_config)
    encoder_path = os.path.join(args.checkpoint_dir, args.encoder_file)
    encoder_state_dict = load_file(encoder_path)
    encoder_state_dict = {k: v.to(torch.bfloat16) for k, v in encoder_state_dict.items()}
    missing, unexpected = points_map_encoder.load_state_dict(encoder_state_dict)
    print(f"Missing keys when loading points map encoder: {missing}")
    print(f"Unexpected keys when loading points map encoder: {unexpected}")
    points_map_encoder.eval()
    points_map_encoder = points_map_encoder.to("cuda")

    # 加载数据
    if args.dataset_type == "drag":
        dataset = DragDataset(jsonl_file=args.dataset_jsonl, root_dir="/mnt/disk1/datasets/drag_data/selectframe")
    elif args.dataset_type == "dragbench":
        dataset = DragBenchDataset(
            root_dir=args.dragbench_root,
            annotation_variant=args.annotation_variant,
            only_annotated=args.only_annotated,
        )
        print(f"[DragBench] variant='{args.annotation_variant}' only_annotated={args.only_annotated} samples={len(dataset)}")

    reverse = bool(args.reverse_direction and args.dataset_type == "drag")
    if args.reverse_direction and not reverse:
        print("[warn] --reverse_direction only applies to --dataset_type drag; ignored.")
    if reverse:
        print("[drag] reverse_direction=True: swapping src<->tgt (image + points) for backward eval")

    for i in range(len(dataset)):
        output_dir = args.output_dir
        prefix = f"{i:04d}_"
        data = dataset[i]

        if args.dataset_type == "drag":
            # DragDataset returns tensors [-1, 1], convert to PIL
            src_image = tensor_to_Image(data["input_image"])
            tgt_image = tensor_to_Image(data["target_image"])
            if reverse:
                src_image, tgt_image = tgt_image, src_image
        else:
            # DragBenchDataset returns PIL images directly, no target_image key
            src_image = data["input_image"]
            tgt_image = data.get("user_drag", None)

        if tgt_image is not None:
            tgt_image.save(os.path.join(output_dir, f"{prefix}gt.jpg"))

        # with open(os.path.join(output_dir, f"{prefix}instruction.txt"), "w") as f:
        #     f.write("Drag the image according to the given points mapping embeddings.")

        # get points_emb
        src_points_np = data["src_points"]
        tgt_points_np = data["tgt_points"]
        if reverse:
            src_points_np, tgt_points_np = tgt_points_np, src_points_np
        if isinstance(src_points_np, torch.Tensor):
            src_points_np = src_points_np.numpy()
        if isinstance(tgt_points_np, torch.Tensor):
            tgt_points_np = tgt_points_np.numpy()
        if args.augment_points:
            W_img, H_img = src_image.size
            src_points_np, tgt_points_np = augment_drag_points(
                src_points_np, tgt_points_np,
                radius=args.augment_radius,
                num_neighbors=args.augment_num,
                img_w=W_img, img_h=H_img,
            )

        src_points = torch.from_numpy(src_points_np).unsqueeze(0)
        tgt_points = torch.from_numpy(tgt_points_np).unsqueeze(0)

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
        save_name = os.path.join(output_dir, f"{prefix}" + ("tgt.jpg" if args.use_lora else "tgt_no_lora.jpg"))
        #image.save(save_name)

        visualize_drag_points(src_image, image, src_points.squeeze(0), tgt_points.squeeze(0), os.path.join(output_dir, f"{prefix}compare.jpg"))

if __name__ == "__main__":
    main()