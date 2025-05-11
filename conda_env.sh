#!/bin/bash

# 配置conda清华源
echo "正在配置conda国内源..."
cat > ~/.condarc << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# 清除缓存
conda clean -i -y

# 检查环境是否存在
if conda info --envs | grep -q svhn_env; then
    echo "环境已存在，正在删除旧环境..."
    conda deactivate
    conda env remove -n svhn_env -y
fi

# 创建conda环境
echo "正在创建conda环境..."
conda create -n svhn_env python=3.8 -y

# 激活环境
source activate svhn_env || source ~/anaconda3/bin/activate svhn_env || conda activate svhn_env

# 验证环境已激活
if [[ "$CONDA_DEFAULT_ENV" != "svhn_env" ]]; then
    echo "环境激活失败，请手动激活环境后继续"
    exit 1
fi

# 安装支持EfficientNet的PyTorch版本 (至少需要1.10版本)
echo "正在安装支持EfficientNet的PyTorch GPU版本..."

# 首先尝试安装最新的PyTorch 1.13版本（支持EfficientNet）
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 验证PyTorch安装
python -c "import torch, torchvision; print('PyTorch版本:',torch.__version__); print('Torchvision版本:',torchvision.__version__); print('CUDA是否可用:',torch.cuda.is_available()); import torchvision.models as models; print('EfficientNet是否可用:', hasattr(models, 'efficientnet_b0'))"

if [ $? -ne 0 ] || ! python -c "import torchvision.models as models; exit(0 if hasattr(models, 'efficientnet_b0') else 1)"; then
    echo "PyTorch 1.13.1安装失败或不支持EfficientNet，尝试使用pip安装PyTorch 1.11.0..."
    conda uninstall -y pytorch torchvision torchaudio
    
    # 配置pip清华源
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 使用pip安装PyTorch 1.11.0（确保支持EfficientNet和CUDA）
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    
    # 再次验证
    python -c "import torch, torchvision; print('PyTorch版本:',torch.__version__); print('Torchvision版本:',torchvision.__version__); print('CUDA是否可用:',torch.cuda.is_available()); import torchvision.models as models; print('EfficientNet是否可用:', hasattr(models, 'efficientnet_b0'))"
    
    if [ $? -ne 0 ] || ! python -c "import torchvision.models as models; exit(0 if hasattr(models, 'efficientnet_b0') else 1)"; then
        echo "PyTorch安装仍然失败或不支持EfficientNet，请手动安装"
        exit 1
    fi
fi

# 检查CUDA是否可用
python -c "import torch; print('CUDA是否可用:',torch.cuda.is_available()); print('GPU信息:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 安装其他依赖
echo "正在安装其他依赖..."
pip install numpy==1.20.3 pandas==1.3.3 pillow==8.3.1 tqdm==4.62.2 matplotlib==3.4.3 scikit-learn==0.24.2 albumentations==1.0.3

echo "环境配置完成！"
echo "PyTorch版本信息:"
python -c "import torch, torchvision; print('PyTorch版本:',torch.__version__); print('Torchvision版本:',torchvision.__version__); print('CUDA是否可用:',torch.cuda.is_available()); print('EfficientNet是否可用:', hasattr(torchvision.models, 'efficientnet_b0'))"
echo "使用 'conda activate svhn_env' 激活环境" 