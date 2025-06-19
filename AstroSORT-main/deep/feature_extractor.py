import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt

from .model import PartBasedTransformer


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        # 使用注意力引导的Transformer模型
        self.net = PartBasedTransformer(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        
        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        if 'net_dict' in state_dict:
            state_dict = state_dict['net_dict']
        self.net.load_state_dict(state_dict)
        
        logger = logging.getLogger("root.tracker")
        logger.info("Loading Attention-Guided Transformer weights from {}... Done!".format(model_path))
        
        self.net.to(self.device)
        self.net.eval()  # 设置为评估模式
        self.size = (64, 128)  # 保持与原始尺寸一致
        
        # 图像归一化与之前保持一致
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        预处理图像:
        1. 转换为0-1的浮点数
        2. 调整大小为(64, 128)
        3. 归一化
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """
        提取特征向量
        """
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
    
    def extract_with_attention(self, im_crop):
        """
        提取特征并返回注意力图，用于可视化
        """
        im_batch = self._preprocess([im_crop])
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            image, attention_maps = self.net.visualize_attention(im_batch)
        
        # 转换为numpy数组
        image = image[0].cpu().permute(1, 2, 0).numpy()
        # 反归一化
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        attention_numpy = [att[0, 0].cpu().numpy() for att in attention_maps]
        
        return image, attention_numpy
    
    def visualize_attention(self, im_crop, save_path=None):
        """
        可视化各部件的注意力图
        """
        image, attention_maps = self.extract_with_attention(im_crop)
        
        # 创建图像网格
        n_parts = len(attention_maps)
        plt.figure(figsize=(12, 4))
        
        # 显示原始图像
        plt.subplot(1, n_parts+1, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示每个部件的注意力图
        for i, att_map in enumerate(attention_maps):
            plt.subplot(1, n_parts+1, i+2)
            plt.imshow(att_map, cmap='jet')
            plt.title(f'Part {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
