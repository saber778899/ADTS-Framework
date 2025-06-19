import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PartBasedTransformer(nn.Module):
    def __init__(self, img_size=(128, 64), embed_dim=128, num_parts=3, 
                 depth=2, num_heads=4, num_classes=751, reid=False):
        super().__init__()
        self.reid = reid
        self.num_parts = num_parts
        
        # 基础特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64x32
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x32x16
            
            nn.Conv2d(128, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True)  # 256x32x16
        )
        
        # 添加部件定位网络 - 注意力引导的自适应划分
        self.part_locator = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.Conv2d(embed_dim, num_parts, 1)  # 每个通道对应一个部件的注意力图
        )
        
        # 为每个部分创建单独的Transformer
        self.part_transformers = nn.ModuleList()
        for _ in range(num_parts):
            encoder_layer = TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            )
            self.part_transformers.append(
                TransformerEncoder(encoder_layer, num_layers=depth)
            )
        
        # 全局特征Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.global_transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 分类器
        self.part_classifiers = nn.ModuleList()
        for _ in range(num_parts):
            self.part_classifiers.append(nn.Linear(embed_dim, num_classes))
        
        self.global_classifier = nn.Linear(embed_dim, num_classes)
        
        # 特征融合层
        self.fusion = nn.Linear(embed_dim * (num_parts + 1), num_classes)
        
        # 添加Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        B = x.shape[0]
        
        # 基础特征提取
        feat = self.features(x)  # B x C x H x W
        
        # 生成部件注意力图
        part_attention = self.part_locator(feat)  # B x num_parts x H x W
        
        # 将注意力图转换为空间上的概率分布（对每个部件，所有位置的权重和为1）
        part_attention = torch.softmax(part_attention.view(B, self.num_parts, -1), dim=2)
        part_attention = part_attention.view(B, self.num_parts, feat.size(2), feat.size(3))
        
        # 使用注意力图加权提取各部件特征
        part_features = []
        
        for i in range(self.num_parts):
            # 应用注意力权重到特征图
            weighted_feat = feat * part_attention[:, i:i+1, :, :]  # B x C x H x W
            
            # 转换为tokens并通过Transformer
            part_tokens = weighted_feat.flatten(2).transpose(1, 2)  # B x (H*W) x C
            part_tokens = self.part_transformers[i](part_tokens)
            
            # 全局池化得到该部件的特征向量
            part_vector = self.dropout(part_tokens.mean(dim=1))  # B x C
            part_features.append(part_vector)
        
        # 全局特征
        global_tokens = feat.flatten(2).transpose(1, 2)  # B x (H*W) x C
        global_tokens = self.global_transformer(global_tokens)
        global_vector = global_tokens.mean(dim=1)  # B x C
        part_features.append(global_vector)
        
        # 特征融合
        all_features = torch.cat(part_features, dim=1)  # B x (num_parts+1)*C
        
        # reid模式
        if self.reid:
            return all_features.div(all_features.norm(p=2, dim=1, keepdim=True))
        
        # 计算各部分和全局的预测
        part_preds = [classifier(feat) for classifier, feat in 
                      zip(self.part_classifiers, part_features[:-1])]
        global_pred = self.global_classifier(global_vector)
        
        # 融合所有特征进行最终预测
        fusion_pred = self.fusion(all_features)
        
        # 训练时返回所有预测（多任务学习），测试时只返回融合预测
        if self.training:
            return part_preds + [global_pred, fusion_pred]
        else:
            return fusion_pred

    def visualize_attention(self, x):
        """
        用于可视化各部件的注意力图
        返回：原始图像和对应的注意力图列表
        """
        B = x.shape[0]
        
        # 基础特征提取
        feat = self.features(x)  # B x C x H x W
        
        # 生成部件注意力图
        part_attention = self.part_locator(feat)  # B x num_parts x H x W
        part_attention = torch.softmax(part_attention.view(B, self.num_parts, -1), dim=2)
        part_attention = part_attention.view(B, self.num_parts, feat.size(2), feat.size(3))
        
        # 将注意力图上采样到原始图像大小
        attention_maps = []
        for i in range(self.num_parts):
            attention_map = nn.functional.interpolate(
                part_attention[:, i:i+1, :, :],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
            attention_maps.append(attention_map)
        
        return x, attention_maps