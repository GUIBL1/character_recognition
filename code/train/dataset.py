import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SVHNDataset(Dataset):
    def __init__(self, img_dir, json_path, transforms=None, max_chars=6):
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_chars = max_chars
        
        # 加载标注数据
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        
        # 获取图片文件名列表
        self.img_names = list(self.labels.keys())
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # 获取标注数据
        label_info = self.labels[img_name]
        
        # 处理多字符标签
        bboxes = []
        labels = []
        
        # 修改数据处理部分，适应实际的JSON结构
        # JSON结构是一个字典，包含'height', 'label', 'left', 'top', 'width'这些字段，每个字段都是一个列表
        for i in range(len(label_info['label'])):
            # 从各个列表中获取对应索引的值
            label = int(label_info['label'][i])
            x = label_info['left'][i]
            y = label_info['top'][i]
            w = label_info['width'][i]
            h = label_info['height'][i]
            
            bboxes.append([x, y, x+w, y+h])
            labels.append(label)
        
        # 确保字符数量不超过最大限制，多余的字符将被截断
        if len(labels) > self.max_chars:
            print(f"警告: 图片 {img_name} 包含 {len(labels)} 个字符，超过最大限制 {self.max_chars}，将截断")
            labels = labels[:self.max_chars]
            bboxes = bboxes[:self.max_chars]
        
        # 确保至少max_chars个字符的长度，不足的用0填充
        while len(labels) < self.max_chars:
            labels.append(0)  # 0表示无字符
        
        # 应用数据增强
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']
        
        return {
            'image': img,
            'labels': torch.tensor(labels, dtype=torch.long),
            'img_name': img_name
        }

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=128, width=128),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=10),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=128, width=128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 