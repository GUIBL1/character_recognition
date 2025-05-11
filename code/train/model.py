import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SVHNModel(nn.Module):
    def __init__(self, num_classes=10, num_chars=6, backbone_type='resnet18'):
        super(SVHNModel, self).__init__()
        
        # 检测是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"模型将使用设备: {self.device}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 检查torchvision版本，选择合适的模型
        torchvision_version = [int(x) for x in torch.__version__.split('.')[:2]]
        supports_efficientnet = False
        if hasattr(models, 'efficientnet_b0'):
            supports_efficientnet = True
            print(f"当前torchvision版本支持EfficientNet")
        else:
            print(f"当前torchvision版本不支持EfficientNet，版本太低，将使用ResNet18替代")
            if backbone_type == 'efficientnet':
                backbone_type = 'resnet18'
                print(f"已自动将backbone从efficientnet切换为resnet18")
        
        # 选择预训练的主干网络
        if backbone_type == 'efficientnet' and supports_efficientnet:
            # 使用getattr动态获取，以兼容不同版本的torchvision
            self.backbone = getattr(models, 'efficientnet_b0')(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            print("使用预训练的EfficientNet-B0作为特征提取器")
        elif backbone_type == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNet34作为特征提取器")
        elif backbone_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNet50作为特征提取器")
        elif backbone_type == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNet101作为特征提取器")
        elif backbone_type == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNet152作为特征提取器")
        elif backbone_type == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNeXt-50作为特征提取器")
        elif backbone_type == 'resnext101':
            self.backbone = models.resnext101_32x8d(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNeXt-101作为特征提取器")
        elif backbone_type == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的Wide ResNet-50作为特征提取器")
        elif backbone_type == 'wide_resnet101':
            self.backbone = models.wide_resnet101_2(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的Wide ResNet-101作为特征提取器")
        elif backbone_type == 'densenet':
            self.backbone = models.densenet121(pretrained=True)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            print("使用预训练的DenseNet121作为特征提取器")
        else:  # 默认使用resnet18
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            print("使用预训练的ResNet18作为特征提取器")
        
        # 打印特征维度
        print(f"特征维度: {in_features}")
        
        # 添加自定义的分类头，为每个字符位置创建一个分类器
        self.num_chars = num_chars
        print(f"支持的最大字符数: {num_chars}")
        self.char_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            ) for _ in range(num_chars)
        ])
        
        # 冻结部分主干网络参数，只训练后面的层
        self._freeze_backbone_layers()
    
    def _freeze_backbone_layers(self):
        """冻结主干网络的前面层，只训练后面的层"""
        # 计算层数的75%作为冻结的界限
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:  # 只训练最后两层
                param.requires_grad = False
        
        # 统计需要训练的参数
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"主干网络参数: {total_params:,}, 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 对每个字符位置进行分类
        logits = [classifier(features) for classifier in self.char_classifiers]
        
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        loss = 0
        for i, input in enumerate(inputs):
            ce_loss = self.ce_loss(input, targets[:, i])
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                focal_loss = focal_loss.mean()
            elif self.reduction == 'sum':
                focal_loss = focal_loss.sum()
            
            loss += focal_loss
        
        return loss / len(inputs) 