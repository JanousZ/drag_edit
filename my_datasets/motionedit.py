import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import math

class MotionEditDataset(Dataset):
    def __init__(self, files_path, is_image_preprocess=True):
        dfs = []
        for file_path in files_path:
            data = pd.read_parquet(file_path)
            dfs.append(data)
        self.data = pd.concat(dfs, ignore_index=True)

        self.resolutions = {}
        for i in range(len(self.data)):
            # 假设存储结构如你所写，我们只读 header 获取 size 以加快速度
            row = self.data.iloc[i]
            with Image.open(BytesIO(row["input_image"]["bytes"])) as img:
                if img.size not in self.resolutions:
                    self.resolutions[img.size] = 0
        
        self.size = 512 * 512
        self.is_image_preprocess = is_image_preprocess

    def __len__(self):
        return len(self.data)

    def image_preprocess(self, image):
        # 1. 获取原始宽高
        w, h = image.size
        
        # 2. 计算缩放比例
        # s = sqrt(目标面积 / 当前面积)
        scale = math.sqrt(self.size / (w * h))
        
        # 3. 计算新的宽高并取整
        new_w = int(round(w * scale / 16.0) * 16)
        new_h = int(round(h * scale / 16.0) * 16)
        
        # 4. 执行缩放 (建议使用 Resampling.LANCZOS 保持质量)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img = np.array(image)
        img = img.astype(np.float32) / 127.5 - 1.0  # -> [-1,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> [C,H,W]
        img = img.to(dtype=torch.float32)
        return img

    def __getitem__(self, index):
        row = self.data.iloc[index]

        raw_input_image = Image.open(BytesIO(row["input_image"]["bytes"]))
        raw_target_image = Image.open(BytesIO(row["target_image"]["bytes"]))

        if self.is_image_preprocess:
            input_image =  self.image_preprocess(raw_input_image)    
            target_image = self.image_preprocess(raw_target_image)
        else:
            input_image = raw_input_image
            target_image = raw_target_image   

        return {
            "prompt": row["prompt"],
            "input_image": input_image, #Image.open(BytesIO(row["input_image"]["bytes"])),
            "target_image": target_image, #Image.open(BytesIO(row["target_image"]["bytes"]))
        }

if __name__ == "__main__":
    files_path = ['train-00000-of-00006.parquet',
                'train-00001-of-00006.parquet',
                'train-00002-of-00006.parquet',
                'train-00003-of-00006.parquet',
                'train-00004-of-00006.parquet',
                'train-00005-of-00006.parquet']
    motionedit_dataset = MotionEditDataset(files_path)

# print(motionedit_dataset[0]["input_image"].shape)



