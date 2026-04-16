import json
import cv2
import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset
import pickle
from PIL import Image

class DragDataset(Dataset):
    def __init__(self, jsonl_file, root_dir):
        self.data = []
        self.crop_size = 512
        self.root_dir = root_dir
        if isinstance(jsonl_file, str):
            jsonl_file = [jsonl_file]
        for jf in jsonl_file:
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("label") == "yes":
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
        rel_folder = record["folder"]
        stride = record["stride"]
        folder_abs = os.path.join(self.root_dir, rel_folder)

        # 优先从 src_points/tgt_points 字段读 npy
        if "src_points" in record and "tgt_points" in record:
            pred_track_path1 = os.path.join(self.root_dir, record["src_points"])
            pred_track_path2 = os.path.join(self.root_dir, record["tgt_points"])
        else:
            # fallback: 从 pair 文件名推断
            rel_path1, rel_path2 = record["pair"]
            frame1 = os.path.basename(rel_path1).split('_frame_')[1].split('.png')[0]
            frame2 = os.path.basename(rel_path2).split('_frame_')[1].split('.png')[0]
            pred_track_path1 = os.path.join(folder_abs, f"pred_track_stride_{stride}_frame_{frame1}.npy")
            if not os.path.exists(pred_track_path1):
                pred_track_path1 = os.path.join(folder_abs, f"pred_track_frame_{frame1}.npy")
            pred_track_path2 = os.path.join(folder_abs, f"pred_track_stride_{stride}_frame_{frame2}.npy")
            if not os.path.exists(pred_track_path2):
                pred_track_path2 = os.path.join(folder_abs, f"pred_track_frame_{frame2}.npy")

        src_points = np.load(pred_track_path1) if os.path.exists(pred_track_path1) else None
        tgt_points = np.load(pred_track_path2) if os.path.exists(pred_track_path2) else None

        # original_frame 路径从 pair 文件名推断
        rel_path1 = record["pair"][0]
        frame1 = os.path.basename(rel_path1).split('_frame_')[1].split('.png')[0]
        rel_path2 = record["pair"][1]
        frame2 = os.path.basename(rel_path2).split('_frame_')[1].split('.png')[0]
        img_path1 = os.path.join(folder_abs, f"original_frame_{frame1}.png")
        img_path2 = os.path.join(folder_abs, f"original_frame_{frame2}.png")

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

class DragBenchDataset(Dataset):
    """
    Loader supporting dragbench-dr (category folders), dragbench-sr (flat samples),
    or both at once.

    Parameters:
    - root_dir: root folder that contains 'dragbench-dr' and/or 'dragbench-sr'
    - bench_type: 'dr', 'sr', 'both', or a list like ['dr','sr']
    - transform: optional callable applied to PIL.Image
    - return_paths: if True, include file paths in returned dict
    """
    def __init__(self, root_dir, bench_type='both', transform=None, return_paths=False, target_size=512):
        if isinstance(bench_type, str):
            bench_type = [bench_type]
        bench_type = set(x.lower() for x in bench_type)
        allowed = {'dr', 'sr', 'both'}
        if 'both' in bench_type:
            bench_type = {'dr', 'sr'}

        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths
        self.samples = []  # list of dicts: {sample_dir, bench_type, category}
        self.target_size = int(target_size)

        if 'dr' in bench_type:
            dr_root = os.path.join(root_dir, 'dragbench-dr')
            if os.path.isdir(dr_root):
                for category in sorted(os.listdir(dr_root)):
                    cat_dir = os.path.join(dr_root, category)
                    if not os.path.isdir(cat_dir):
                        continue
                    for sample in sorted(os.listdir(cat_dir)):
                        sample_dir = os.path.join(cat_dir, sample)
                        if os.path.isdir(sample_dir):
                            self.samples.append({
                                'sample_dir': sample_dir,
                                'bench_type': 'dr',
                                'category': category
                            })

        if 'sr' in bench_type:
            sr_root = os.path.join(root_dir, 'dragbench-sr')
            if os.path.isdir(sr_root):
                for sample in sorted(os.listdir(sr_root)):
                    sample_dir = os.path.join(sr_root, sample)
                    if os.path.isdir(sample_dir):
                        self.samples.append({
                            'sample_dir': sample_dir,
                            'bench_type': 'sr',
                            'category': None
                        })

        if len(self.samples) == 0:
            raise ValueError(f"No samples found under {root_dir} for types {bench_type}")

    def __len__(self):
        return len(self.samples)

    def _open_image(self, path):
        if not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")

    def _load_pickle(self, path):
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _compute_points_arrays(self, meta):
        """
        From meta_data['points'] (assumed [src0, tgt0, src1, tgt1, ...]) return src_pts, tgt_pts arrays.
        If missing or invalid, return (None, None).
        """
        if meta is None:
            return None, None
        pts = meta.get("points")
        if not pts:
            return None, None
        try:
            src_pts = np.array(pts[0::2], dtype=np.float32)
            tgt_pts = np.array(pts[1::2], dtype=np.float32)
            return src_pts, tgt_pts
        except Exception:
            return None, None

    def _crop_resize_and_update_points(self, img, src_pts, tgt_pts):
        """
        Crop img to square ROI covering points (or center square if points None),
        resize to target_size if needed, and return (img_out, new_src, new_tgt, crop_info)
        crop_info = (x0, y0, side, scale)
        """
        w, h = img.size
        t = self.target_size

        # if already desired size return without change
        if w == t and h == t:
            return img, src_pts, tgt_pts, (0, 0, max(w, h), 1.0)

        # determine ROI
        if src_pts is not None and tgt_pts is not None and len(src_pts) > 0 and len(tgt_pts) > 0:
            all_pts = np.vstack([src_pts, tgt_pts])
            min_x, min_y = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            max_x, max_y = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
            content_w = max_x - min_x
            content_h = max_y - min_y
            base_side = max(content_w, content_h, t)
            base_side = min(base_side, w, h)
            center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            x0 = int(round(center_x - base_side / 2.0))
            y0 = int(round(center_y - base_side / 2.0))
        else:
            # center square fallback
            base_side = min(w, h, max(t, 1))
            x0 = (w - base_side) // 2
            y0 = (h - base_side) // 2
            base_side = int(base_side)

        x0 = max(0, min(x0, w - int(base_side)))
        y0 = max(0, min(y0, h - int(base_side)))
        actual_side = int(base_side)

        crop_box = (x0, y0, x0 + actual_side, y0 + actual_side)
        cropped = img.crop(crop_box)

        scale = 1.0
        if actual_side != t:
            cropped = cropped.resize((t, t), Image.LANCZOS)
            scale = float(t) / float(actual_side)

        new_src, new_tgt = None, None
        if src_pts is not None:
            new_src = (src_pts - np.array([x0, y0])) * scale
        if tgt_pts is not None:
            new_tgt = (tgt_pts - np.array([x0, y0])) * scale

        return cropped, new_src, new_tgt, (x0, y0, actual_side, scale)

    def __getitem__(self, idx):
        info = self.samples[idx]
        sample_dir = info['sample_dir']

        orig_path = os.path.join(sample_dir, "original_image.png")
        user_drag_path = os.path.join(sample_dir, "user_drag.png")
        meta_path = os.path.join(sample_dir, "meta_data.pkl")

        original_image = self._open_image(orig_path)
        user_drag = self._open_image(user_drag_path)
        meta_data = self._load_pickle(meta_path)

        # optional i4p files
        meta_i4p = self._load_pickle(os.path.join(sample_dir, "meta_data_i4p.pkl"))
        user_drag_i4p = self._open_image(os.path.join(sample_dir, "user_drag_i4p.png"))

        # get point arrays if present
        src_points, tgt_points = self._compute_points_arrays(meta_data)

        # crop/resize original_image and update points
        if original_image is not None:
            original_image, new_src, new_tgt, crop_info = self._crop_resize_and_update_points(original_image, src_points, tgt_points)
        else:
            new_src, new_tgt = src_points, tgt_points

        # apply same crop/resize to user_drag and user_drag_i4p (use same logic)
        if user_drag is not None:
            user_drag, _, _, _ = self._crop_resize_and_update_points(user_drag, src_points, tgt_points)
        if user_drag_i4p is not None:
            user_drag_i4p, _, _, _ = self._crop_resize_and_update_points(user_drag_i4p, src_points, tgt_points)

        # write updated points back into meta_data["points"] if meta exists
        if meta_data is not None and new_src is not None and new_tgt is not None:
            interleaved = []
            for s, tpt in zip(new_src.tolist(), new_tgt.tolist()):
                interleaved.append([float(s[0]), float(s[1])])
                interleaved.append([float(tpt[0]), float(tpt[1])])
            meta_data["points"] = interleaved
            src_points = new_src
            tgt_points = new_tgt
        else:
            src_points = new_src
            tgt_points = new_tgt

        # apply transforms if provided (after crop/resize)
        if self.transform is not None:
            original_image = self.transform(original_image) if original_image is not None else None
            user_drag = self.transform(user_drag) if user_drag is not None else None
            if user_drag_i4p is not None:
                user_drag_i4p = self.transform(user_drag_i4p)

        out = {
            "input_image": original_image,
            "user_drag": user_drag,
            "meta_data": meta_data,
            "meta_data_i4p": meta_i4p,
            "user_drag_i4p": user_drag_i4p,
            "sample_dir": sample_dir,
            "bench_type": info['bench_type'],
            "category": info['category'],
            "src_points": src_points,
            "tgt_points": tgt_points,
        }
        if self.return_paths:
            out.update({
                "original_image_path": orig_path,
                "user_drag_path": user_drag_path,
                "meta_data_path": meta_path,
            })
        return out

if __name__ == "__main__":
    bench = DragBenchDataset(root_dir = "/mnt/disk1/datasets/DragBench")
