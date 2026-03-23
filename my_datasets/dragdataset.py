import json
import cv2
import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset

class DragDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        self.crop_size = 512
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    self.data.append(record)
                except json.JSONDecodeError:
                    continue

    def image_preprocess(self, img_path1, img_path2, src_points, tgt_points):
        # 加载图像
        src_image = cv2.imread(img_path1) # H, W, C (BGR)
        tgt_image = cv2.imread(img_path2)
        h_orig, w_orig, _ = src_image.shape

        # 1. 计算点集的边界
        all_points = np.concatenate([src_points, tgt_points], axis=0)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        # 2. 确定需要涵盖的最小正方形区域 (ROI)
        content_w = max_x - min_x
        content_h = max_y - min_y
        base_side = max(content_w, content_h, self.crop_size)
        
        # 限制 base_side 不能超过原图的短边（防止缩放崩溃）
        base_side = min(base_side, h_orig, w_orig)

        # 3. 确定 ROI 的中心并修正偏移量
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        x_offset = int(center_x - base_side / 2)
        y_offset = int(center_y - base_side / 2)

        # 边界修正：确保 ROI 在原图内
        x_offset = max(0, min(x_offset, w_orig - int(base_side)))
        y_offset = max(0, min(y_offset, h_orig - int(base_side)))
        actual_side = int(base_side)

        # 4. 裁剪并缩放到 512x512
        src_crop = src_image[y_offset : y_offset + actual_side, x_offset : x_offset + actual_side]
        tgt_crop = tgt_image[y_offset : y_offset + actual_side, x_offset : x_offset + actual_side]
        
        # 如果实际裁剪的尺寸不是 512，则进行缩放
        scale = self.crop_size / actual_side
        if actual_side != self.crop_size:
            src_crop = cv2.resize(src_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            tgt_crop = cv2.resize(tgt_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # 5. 更新坐标：(原始坐标 - 偏移) * 缩放比例
        new_src_points = (src_points - np.array([x_offset, y_offset])) * scale
        new_tgt_points = (tgt_points - np.array([x_offset, y_offset])) * scale

        # 6. 转换为 Torch Tensor 格式 [-1, 1]
        # 转换为 RGB -> 归一化到 [0, 1] -> 变换到 [-1, 1]
        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
            img_tensor = torch.from_numpy(img).float() # [H, W, C]
            img_tensor = img_tensor.permute(2, 0, 1)  #[c,h,w]
            img_tensor = (img_tensor / 127.5) - 1.0    # 归一化到 [-1, 1]
            return img_tensor

        src_tensor = to_tensor(src_crop)
        tgt_tensor = to_tensor(tgt_crop)

        return src_tensor, new_src_points, tgt_tensor, new_tgt_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("索引超出范围")
        
        record = self.data[idx]
        dir = record["folder"]
        stride = record["stride"]
        img_path1, img_path2 = record["pair"]
        label = record["label"]

        frame1 = os.path.basename(img_path1).split('_frame_')[1].split('.png')[0]
        frame2 = os.path.basename(img_path2).split('_frame_')[1].split('.png')[0]
        pred_track_path1 = os.path.join(dir, f"pred_track_frame_{frame1}.npy")
        pred_track_path2 = os.path.join(dir, f"pred_track_frame_{frame2}.npy")
        src_points = np.load(pred_track_path1) if os.path.exists(pred_track_path1) else None
        tgt_points = np.load(pred_track_path2) if os.path.exists(pred_track_path2) else None
        img_path1 = os.path.join(dir, f"original_frame_{frame1}.png")
        img_path2 = os.path.join(dir, f"original_frame_{frame2}.png")

        src_image, src_points, tgt_image, tgt_points = self.image_preprocess(img_path1, img_path2, src_points, tgt_points)

        item = {
            "input_image": src_image,
            "target_image": tgt_image,
            "src_points": src_points,
            "tgt_points": tgt_points,
        }
        return item

def dd_collate_fn(batch):
    input_images = torch.stack([item['input_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    src_points = torch.stack([torch.as_tensor(item['src_points']) for item in batch])
    tgt_points = torch.stack([torch.as_tensor(item['tgt_points']) for item in batch])
    
    return {
        "input_image": input_images,
        "target_image": target_images,
        "src_points": src_points,
        "tgt_points": tgt_points,
    }