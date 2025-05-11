# 街景门牌号识别项目

## 项目介绍
本项目是基于SVHN（Street View House Numbers）数据集的门牌号识别解决方案。任务是识别图片中的门牌号码，每个图片包含一个或多个数字。

## 算法介绍
本解决方案使用深度学习方法进行门牌号码识别，主要包括以下几个部分：

1. **数据预处理**：
   - 对图像进行尺寸调整、归一化
   - 使用数据增强技术（亮度对比度调整、水平翻转、旋转等）增加模型鲁棒性

2. **模型架构**：
   - 支持多种预训练网络作为特征提取器：
     - EfficientNet-B0
     - ResNet系列 (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
     - ResNeXt系列 (ResNeXt50, ResNeXt101)
     - Wide ResNet系列 (Wide ResNet50, Wide ResNet101)
     - DenseNet121
   - 为每个字符位置添加独立的分类头，支持最多6个字符识别
   - 使用Focal Loss作为损失函数，更好地处理类别不平衡问题
   - 实现了特征提取器参数部分冻结，只训练后面的层，提高训练效率

3. **训练策略**：
   - 支持多种优化器：Adam、AdamW、SGD
   - 支持多种学习率调度策略：余弦退火、OneCycleLR、ReduceLROnPlateau
   - 保存验证集上性能最佳的模型
   - 支持GPU加速训练

## 项目结构
```
project/
│
├── README.md               # 项目说明文件
├── requirements.txt        # Python依赖包列表
├── run.sh                  # 一键运行脚本
├── run_pretrained.sh       # 预训练模型训练脚本
├── conda_env.sh            # Conda环境配置脚本（GPU支持）
│
├── tcdata/                 # 数据目录
│   ├── images/
│   │   └── extracted/      # 解压后的图片
│   │       ├── mchar_test_a/
│   │       ├── mchar_train/
│   │       └── mchar_val/
│   └── json/
│       ├── mchar_train.json
│       └── mchar_val.json
│
├── code/
│   ├── train/              # 训练代码
│   │   ├── dataset.py      # 数据集加载
│   │   ├── model.py        # 模型定义
│   │   └── train.py        # 训练脚本
│   └── test/               # 测试代码
│       └── inference.py    # 预测脚本
│
└── output/                 # 模型输出目录
    ├── best_model.pth      # 最佳模型
    ├── final_model.pth     # 最终模型
    ├── training_log.txt    # 训练日志
    ├── training_plot.png   # 训练过程可视化
    ├── inference_stats.json # 推理统计信息
    └── results_summary.txt # 不同模型结果汇总
```

## 环境配置

### GPU环境（推荐）
使用`conda_env.sh`脚本自动配置支持GPU的环境：
```bash
bash conda_env.sh
```

该脚本会自动：
1. 配置conda清华源
2. 创建conda虚拟环境
3. 安装支持CUDA的PyTorch版本
4. 安装所有依赖包
5. 验证PyTorch安装是否成功

### 依赖包列表
```
torch==1.9.0
torchvision==0.10.0
cudatoolkit=11.1 (GPU版本需要)
numpy==1.20.3
pandas==1.3.3
pillow==8.3.1
tqdm==4.62.2
matplotlib==3.4.3
scikit-learn==0.24.2
albumentations==1.0.3
```

## 使用方法

### 一键运行（基础版）
1. 确保数据放置在正确的位置
2. 运行以下命令：
```bash
bash run.sh
```
这将自动检查和创建conda环境、安装依赖、训练模型并生成预测结果。

### 使用预训练模型（推荐）
使用以下命令运行多个预训练模型并比较结果：
```bash
bash run_pretrained.sh
```

这将会：
1. 对多个预训练模型（EfficientNet、ResNet系列、ResNeXt系列、Wide ResNet系列、DenseNet）进行训练和测试
2. 对每个模型尝试不同的优化器和学习率调度策略
3. 自动生成各个模型的训练曲线和预测结果
4. 在`output/results_summary.txt`中汇总所有模型的性能

如果只想训练特定的模型，可以指定模型名称：
```bash
bash run_pretrained.sh efficientnet
```

### 分步运行
1. **创建环境并安装依赖**：
```bash
bash conda_env.sh
conda activate svhn_env
```

2. **训练特定的预训练模型**：
```bash
cd code/train
python train.py --data_dir ../../tcdata/images/extracted --json_dir ../../tcdata/json --batch_size 64 --epochs 20 --backbone efficientnet --optimizer adamw --scheduler cosine --max_chars 6
```

3. **生成预测结果**：
```bash
cd ../test
python inference.py --test_dir ../../tcdata/images/extracted/mchar_test_a --model_path ../../output/best_model.pth --backbone efficientnet --max_chars 6
```

### 支持的预训练模型
- `efficientnet`：EfficientNet-B0（默认，推荐）
- `resnet18`：ResNet18
- `resnet34`：ResNet34
- `resnet50`：ResNet50
- `resnet101`：ResNet101
- `resnet152`：ResNet152
- `resnext50`：ResNeXt-50-32x4d
- `resnext101`：ResNeXt-101-32x8d
- `wide_resnet50`：Wide ResNet-50-2
- `wide_resnet101`：Wide ResNet-101-2
- `densenet`：DenseNet121

### 支持的优化器
- `adam`：Adam优化器（默认）
- `adamw`：AdamW优化器（推荐）
- `sgd`：SGD优化器（动量0.9）

### 支持的学习率调度器
- `cosine`：余弦退火调度器（默认）
- `onecycle`：OneCycleLR调度器
- `plateau`：ReduceLROnPlateau调度器

### 字符识别参数
- `--max_chars`：指定最大字符数，默认为6

## 模型比较与选择
通过`run_pretrained.sh`脚本，我们可以比较不同模型在以下几个方面的表现：
1. 验证集上的准确率（字符级和序列级）
2. 推理速度（每张图片的平均处理时间）
3. 训练收敛速度（训练曲线）
4. 参数量和计算复杂度

最优的模型配置会根据具体硬件和精度要求来选择，一般来说：
- 对于高精度需求：推荐EfficientNet、ResNeXt或Wide ResNet + AdamW + 余弦退火调度器
- 对于速度优先：推荐ResNet18/34 + Adam + OneCycleLR调度器
- 对于速度和精度的平衡：ResNet50或DenseNet121 + Adam + 余弦退火调度器

## 性能指标
- 训练集大小：3万张图片
- 验证集大小：1万张图片
- 测试集A大小：4万张图片
- 测试集B大小：4万张图片
- 主要评估指标：字符级准确率和序列完全匹配准确率
- 支持的最大字符数：6个
- GPU训练速度比CPU提升约10-20倍

## 注意事项
- 确保PyTorch正确安装，若出现`ModuleNotFoundError: No module named 'torch'`错误，请重新运行`conda_env.sh`脚本
- 若字符数超过最大限制(6个)，会自动截断并发出警告
- 训练和测试过程中会生成详细的日志和统计信息，方便调试和优化
- ResNeXt和Wide ResNet等大模型可能需要更多的GPU内存，请根据硬件条件选择合适的batch_size
