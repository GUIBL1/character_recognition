import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys
import time
import json

# 添加训练代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import SVHNModel

class SVHNTestDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # 获取图片文件名列表
        self.img_names = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # 应用数据增强
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']
        
        return {
            'image': img,
            'img_name': img_name
        }

def get_test_transforms():
    return A.Compose([
        A.Resize(height=128, width=128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='../../tcdata/images/extracted/mchar_test_a')
    parser.add_argument('--model_path', type=str, default='../../output/best_model.pth')
    parser.add_argument('--config_path', type=str, default='../../output/config.txt')
    parser.add_argument('--output_file', type=str, default='../../submit.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU ID，默认为0')
    parser.add_argument('--backbone', type=str, default='efficientnet', 
                        choices=['resnet18', 'resnet50', 'efficientnet', 'densenet'], 
                        help='使用的主干网络，如果不指定将从配置文件读取')
    parser.add_argument('--max_chars', type=int, default=6,
                        help='最大字符数量')
    args = parser.parse_args()
    
    # 尝试从配置文件读取主干网络类型和最大字符数
    backbone_type = args.backbone
    max_chars = args.max_chars
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            for line in f:
                if line.startswith('Backbone:'):
                    backbone_type = line.split(':')[1].strip()
                elif line.startswith('Max Chars:'):
                    try:
                        max_chars = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass  # 如果转换失败，保持默认值
        print(f"从配置文件读取主干网络类型: {backbone_type}")
        print(f"从配置文件读取最大字符数: {max_chars}")
    else:
        print(f"配置文件不存在，使用命令行参数指定的主干网络: {backbone_type}")
        print(f"配置文件不存在，使用命令行参数指定的最大字符数: {max_chars}")
    
    # 设置使用的GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(args.gpu_id)}')
        print(f'可用GPU数量: {torch.cuda.device_count()}')
    
    # 创建数据集和加载器
    test_dataset = SVHNTestDataset(
        img_dir=args.test_dir,
        transforms=get_test_transforms()
    )
    
    print(f'测试集大小: {len(test_dataset)}')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 加载模型
    model = SVHNModel(num_classes=10, num_chars=max_chars, backbone_type=backbone_type)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f'模型已加载, 参数数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'使用主干网络: {backbone_type}')
    print(f'支持的最大字符数: {max_chars}')
    
    # 进行预测
    predictions = []
    img_names = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            images = batch['image'].to(device)
            batch_img_names = batch['img_name']
            
            outputs = model(images)
            
            # 获取每个位置的预测结果
            char_preds = []
            for output in outputs:
                pred = torch.argmax(output, dim=1).cpu().numpy()
                char_preds.append(pred)
            
            # 将预测结果转换为正确的格式
            char_preds = np.array(char_preds).T
            
            # 将数字序列转换为字符串
            for i in range(len(batch_img_names)):
                digits = char_preds[i]
                # 去除填充的0
                digits = [str(d) for d in digits if d != 0]
                # 拼接数字
                label = ''.join(digits)
                
                predictions.append(label)
                img_names.append(batch_img_names[i])
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 创建提交文件
    submission = pd.DataFrame({
        'file_name': img_names,
        'file_code': predictions
    })
    
    submission.to_csv(args.output_file, index=False)
    
    print(f'预测完成! 耗时: {inference_time:.2f}秒')
    print(f'平均每张图片推理时间: {inference_time/len(test_dataset)*1000:.2f}毫秒')
    print(f'提交文件已保存至: {args.output_file}')
    
    # 保存推理统计信息
    stats_file = os.path.join(os.path.dirname(args.output_file), 'inference_stats.json')
    stats = {
        'backbone': backbone_type,
        'max_chars': max_chars,
        'test_samples': len(test_dataset),
        'total_time_seconds': inference_time,
        'avg_time_per_image_ms': inference_time/len(test_dataset)*1000,
        'batch_size': args.batch_size,
        'device': str(device),
    }
    if torch.cuda.is_available():
        stats['gpu_model'] = torch.cuda.get_device_name(args.gpu_id)
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f'推理统计信息已保存至: {stats_file}')
    
    # 打印部分预测结果示例
    print("\n预测结果示例:")
    for i in range(min(5, len(img_names))):
        print(f"{img_names[i]}: {predictions[i]}")
    
    # 分析预测结果
    lengths = [len(p) for p in predictions]
    print(f"\n预测结果长度统计:")
    for length in range(1, max_chars+1):
        count = lengths.count(length)
        percent = count / len(lengths) * 100
        print(f"{length}位数字: {count}个 ({percent:.2f}%)")

if __name__ == '__main__':
    main() 