import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from dataset import SVHNDataset, get_transforms
from model import SVHNModel, FocalLoss

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 如果使用OneCycleLR，需要在每个batch后调整学习率
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f'Train Loss: {total_loss/(pbar.n+1):.4f}, LR: {current_lr:.6f}')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch['image'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 获取每个位置的预测结果
            predictions = []
            for output in outputs:
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred)
            
            # 将预测结果转换为正确的格式
            batch_preds = np.array(predictions).T
            all_predictions.append(batch_preds)
            
            # 保存目标值
            all_targets.append(targets.cpu().numpy())
    
    # 合并所有批次的结果
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # 计算准确率，按照字符位置
    accuracies = []
    for i in range(all_predictions.shape[1]):
        acc = accuracy_score(all_targets[:, i], all_predictions[:, i])
        accuracies.append(acc)
    
    # 计算序列完全匹配的准确率
    seq_accuracy = np.mean(np.all(all_predictions == all_targets, axis=1))
    
    return {
        'loss': total_loss / len(dataloader),
        'char_accuracies': accuracies,
        'avg_char_accuracy': np.mean(accuracies),
        'seq_accuracy': seq_accuracy
    }

def plot_training_results(log_file, output_dir):
    """绘制训练过程的损失和准确率曲线"""
    data = pd.read_csv(log_file)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(data['Epoch'], data['Train Loss'], 'b-', label='Train Loss')
    ax1.plot(data['Epoch'], data['Val Loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(data['Epoch'], data['Val Avg Char Accuracy'], 'g-', label='Char Accuracy')
    ax2.plot(data['Epoch'], data['Val Sequence Accuracy'], 'p-', label='Sequence Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../tcdata/images/extracted')
    parser.add_argument('--json_dir', type=str, default='../../tcdata/json')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='../../output')
    # 添加CUDA相关参数
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU ID，默认为0')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # 添加模型和训练相关参数
    parser.add_argument('--backbone', type=str, default='efficientnet', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
                                'resnext50', 'resnext101', 'wide_resnet50', 'wide_resnet101',
                                'efficientnet', 'densenet'], 
                        help='使用的主干网络')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'adamw'], 
                        help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'onecycle', 'plateau'], 
                        help='学习率调度器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='权重衰减参数')
    parser.add_argument('--max_chars', type=int, default=6, 
                        help='最大字符数量')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置使用的GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(args.gpu_id)}')
        print(f'GPU显存: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024 / 1024 / 1024:.2f} GB')
    
    # 创建数据集和加载器
    print(f"最大字符数: {args.max_chars}")
    train_dataset = SVHNDataset(
        img_dir=os.path.join(args.data_dir, 'mchar_train'),
        json_path=os.path.join(args.json_dir, 'mchar_train.json'),
        transforms=get_transforms(is_train=True),
        max_chars=args.max_chars
    )
    
    val_dataset = SVHNDataset(
        img_dir=os.path.join(args.data_dir, 'mchar_val'),
        json_path=os.path.join(args.json_dir, 'mchar_val.json'),
        transforms=get_transforms(is_train=False),
        max_chars=args.max_chars
    )
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    print(f"使用 {args.backbone} 作为主干网络")
    model = SVHNModel(num_classes=10, num_chars=args.max_chars, backbone_type=args.backbone)
    model = model.to(device)
    
    # 打印模型结构
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数: {total_params:,}')
    print(f'可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)')
    
    # 定义损失函数
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # 选择优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"使用Adam优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        print(f"使用SGD优化器，学习率={args.lr}，动量=0.9，权重衰减={args.weight_decay}")
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"使用AdamW优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    
    # 选择学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
        print(f"使用CosineAnnealingLR调度器，T_max={args.epochs}，eta_min={args.lr/100}")
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr*10, epochs=args.epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=10
        )
        print(f"使用OneCycleLR调度器，max_lr={args.lr*10}，div_factor=10")
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        print(f"使用ReduceLROnPlateau调度器，factor=0.5，patience=2")
    
    # 训练日志文件
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Epoch,Train Loss,Val Loss,Val Avg Char Accuracy,Val Sequence Accuracy\n')
    
    # 保存训练配置
    config_file = os.path.join(args.output_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Max Chars: {args.max_chars}\n")
    
    # 开始训练
    best_accuracy = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            scheduler if args.scheduler == 'onecycle' else None
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 学习率调度（对于非OneCycleLR的调度器）
        if args.scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"当前学习率: {current_lr:.6f}")
        elif args.scheduler == 'plateau':
            scheduler.step(val_metrics['seq_accuracy'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")
        
        epoch_end = time.time()
        epoch_mins, epoch_secs = divmod(epoch_end - epoch_start, 60)
        
        # 打印验证结果
        print(f'Epoch: {epoch+1:02} | 耗时: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Char Accuracies: {[f"{acc:.4f}" for acc in val_metrics["char_accuracies"]]}')
        print(f'Val Avg Char Accuracy: {val_metrics["avg_char_accuracy"]:.4f}')
        print(f'Val Sequence Accuracy: {val_metrics["seq_accuracy"]:.4f}')
        
        # 记录日志
        with open(log_file, 'a') as f:
            f.write(f'{epoch+1},{train_loss:.6f},{val_metrics["loss"]:.6f},{val_metrics["avg_char_accuracy"]:.6f},{val_metrics["seq_accuracy"]:.6f}\n')
        
        # 保存最佳模型
        if val_metrics['seq_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['seq_accuracy']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f'保存最佳模型，序列准确率: {best_accuracy:.4f}')
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    end_time = time.time()
    total_mins, total_secs = divmod(end_time - start_time, 60)
    total_hours, total_mins = divmod(total_mins, 60)
    
    print(f'训练完成！总耗时: {int(total_hours)}h {int(total_mins)}m {total_secs:.2f}s')
    print(f'最佳序列准确率: {best_accuracy:.4f}')
    
    # 绘制训练结果
    plot_training_results(log_file, args.output_dir)
    print(f'训练曲线已保存至: {os.path.join(args.output_dir, "training_plot.png")}')

if __name__ == '__main__':
    main() 