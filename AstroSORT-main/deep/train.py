import argparse
import os
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import PartBasedTransformer  # 导入修改后的模型

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='/home/liuhaoan/deepsort-inhanced/boxmot-3.0/deep_sort_pytorch/deep_sort/deep/Market-1501', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=3, type=int)
parser.add_argument("--lr", default=0.0002, type=float)  # 更小的学习率
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--epochs', default=18, type=int)  # 更多的训练轮数
parser.add_argument('--num-parts', default=3, type=int)  # 部件数量参数化
parser.add_argument('--visualize-attention', action='store_true')  # 添加可视化开关
args = parser.parse_args()

# 创建runs目录（如果不存在）
runs_dir = "checkpoint"
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

# 自动创建实验目录（exp1, exp2, ...）在runs目录下
existing_exps = glob.glob(os.path.join(runs_dir, "exp*"))
exp_nums = [int(os.path.basename(exp).replace("exp", "")) for exp in existing_exps 
            if os.path.basename(exp).replace("exp", "").isdigit()]
next_exp_num = 1 if not exp_nums else max(exp_nums) + 1
exp_dir = os.path.join(runs_dir, f"exp{next_exp_num}")

# 创建实验目录及其checkpoint子目录
exp_checkpoint_dir = os.path.join(exp_dir, "checkpoint")
vis_dir = os.path.join(exp_dir, "attention_vis")  # 添加注意力可视化目录
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(exp_checkpoint_dir):
    os.makedirs(exp_checkpoint_dir)
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

print(f"实验将保存在目录: {exp_dir}")

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# 增强数据加载和增强
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")

# 更强的数据增强
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=10),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    torchvision.transforms.RandomErasing(p=0.5)
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64,  # 更小的批次大小
    shuffle=True,
    num_workers=8  # 利用多线程加载
)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64,
    shuffle=True,
    num_workers=8
)

num_classes = max(len(trainloader.dataset.classes),
                  len(testloader.dataset.classes))

# 创建PartBasedTransformer模型，现在使用注意力引导的划分
start_epoch = 0
net = PartBasedTransformer(
    img_size=(128, 64),
    embed_dim=128,
    num_parts=args.num_parts,  # 使用命令行参数
    depth=2,
    num_heads=4,
    num_classes=num_classes,
    reid=False  # 训练时为False
)

if args.resume:
    checkpoint_path = "./checkpoint/ckpt.t7"
    assert os.path.isfile(checkpoint_path), f"Error: no checkpoint file found at {checkpoint_path}!"
    print(f'Loading from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# 使用AdamW优化器和余弦退火学习率调度器
optimizer = torch.optim.AdamW(
    net.parameters(), 
    lr=args.lr, 
    weight_decay=1e-4  # 适中的权重衰减
)

# 余弦退火学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs
)

# 使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
best_acc = 0.

# 添加注意力正则化损失函数，鼓励不同部件关注不同区域
def attention_diversity_loss(attention_maps):
    """计算注意力多样性损失，鼓励不同部件关注不同区域"""
    batch_size, num_parts, h, w = attention_maps.size()
    attention_flat = attention_maps.view(batch_size, num_parts, -1)
    
    # 计算部件之间的相似度矩阵
    similarity = torch.bmm(attention_flat, attention_flat.transpose(1, 2))
    
    # 对角线元素应该接近1（自己和自己的相似度），非对角元素应该接近0（不同部件不重叠）
    # 创建目标矩阵（对角线为1，其他为0）
    target = torch.eye(num_parts, device=attention_maps.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # 计算相似度矩阵与目标矩阵之间的均方误差
    diversity_loss = torch.nn.functional.mse_loss(similarity, target)
    
    return diversity_loss

# 可视化注意力图函数
def visualize_attention_maps(images, attention_maps, epoch, batch_idx, save_dir):
    """保存注意力图可视化结果"""
    if batch_idx % 50 != 0:  # 每50个批次可视化一次
        return
        
    # 只处理批次中的第一张图像
    image = images[0].cpu().permute(1, 2, 0).numpy()
    # 反归一化
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    # 创建可视化图
    num_parts = attention_maps.size(1)
    fig, axes = plt.subplots(1, num_parts + 1, figsize=(4 * (num_parts + 1), 6))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 注意力图
    for i in range(num_parts):
        att = attention_maps[0, i].detach().cpu().numpy()
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(att, alpha=0.5, cmap='jet')
        axes[i + 1].set_title(f'Part {i+1} Attention')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'attention_e{epoch}_b{batch_idx}.png')
    plt.savefig(save_path)
    plt.close(fig)

# 训练函数
def train(epoch):
    print("\nEpoch : %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 获取基础特征和注意力图
        feat = net.features(inputs)
        part_attention = net.part_locator(feat)
        part_attention = torch.softmax(part_attention.view(inputs.size(0), net.num_parts, -1), dim=2)
        part_attention = part_attention.view(inputs.size(0), net.num_parts, feat.size(2), feat.size(3))
        
        # 前向传播
        outputs = net(inputs)
        
        # 由于Part-based模型返回多个分类结果，需要特殊处理
        if isinstance(outputs, list):
            # 计算每个部分分类器的损失和融合分类器的损失
            loss = 0
            for part_pred in outputs[:-1]:  # 所有分类器预测，最后一个是融合预测
                loss += criterion(part_pred, labels)
            
            # 融合分类器损失权重加大
            fusion_pred = outputs[-1]
            loss += 2.0 * criterion(fusion_pred, labels)
            
            # 添加注意力多样性损失
            diversity_loss = attention_diversity_loss(part_attention)
            loss += 0.1 * diversity_loss  # 权重系数可调
            
            # 单独记录融合分类器的损失
            fusion_loss = criterion(fusion_pred, labels)

            # 计算正确率时使用融合预测
            pred = fusion_pred.max(dim=1)[1]
        else:
            # 单一预测情况
            loss = criterion(outputs, labels)
            fusion_loss = loss
            pred = outputs.max(dim=1)[1]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 可视化注意力图（如果启用）
        if args.visualize_attention and idx % 50 == 0:  # 每50个批次可视化一次
            # 将注意力图上采样到原始图像大小用于可视化
            upsampled_attention = torch.nn.functional.interpolate(
                part_attention, 
                size=(inputs.size(2), inputs.size(3)), 
                mode='bilinear',
                align_corners=False
            )
            visualize_attention_maps(inputs, upsampled_attention, epoch, idx, vis_dir)

        # 累计损失和准确率
        training_loss += fusion_loss.item()
        train_loss += fusion_loss.item()
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # 打印进度
        if (idx+1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss /
                interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss/len(trainloader), 1. - correct/total

# 测试函数
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            
            # 同样处理多部分输出
            if isinstance(outputs, list):
                # 使用融合分类器的输出计算损失
                loss = criterion(outputs[-1], labels)
                pred = outputs[-1].max(dim=1)[1]
            else:
                loss = criterion(outputs, labels)
                pred = outputs.max(dim=1)[1]

            test_loss += loss.item()
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100.*(idx+1)/len(testloader), end-start, test_loss /
            len(testloader), correct, total, 100.*correct/total
        ))

    # 保存最佳模型
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        best_ckpt_path = os.path.join(exp_checkpoint_dir, "ckpt.t7")
        print(f"保存参数到 {best_ckpt_path}")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(checkpoint, best_ckpt_path)
        
        # 在测试集上可视化最佳模型的注意力图
        if args.visualize_attention:
            visualize_best_attention(inputs, epoch)

    return test_loss/len(testloader), 1. - correct/total

# 可视化最佳模型的注意力图
def visualize_best_attention(test_images, epoch):
    """为测试集中的几张图像可视化最佳模型的注意力分配"""
    # 选择前8张图像
    sample_images = test_images[:8].to(device)
    
    # 获取基础特征和注意力图
    with torch.no_grad():
        feat = net.features(sample_images)
        part_attention = net.part_locator(feat)
        part_attention = torch.softmax(part_attention.view(sample_images.size(0), net.num_parts, -1), dim=2)
        part_attention = part_attention.view(sample_images.size(0), net.num_parts, feat.size(2), feat.size(3))
        
        # 上采样到原始图像大小
        upsampled_attention = torch.nn.functional.interpolate(
            part_attention, 
            size=(sample_images.size(2), sample_images.size(3)), 
            mode='bilinear',
            align_corners=False
        )
    
    # 为每个样本创建可视化
    for i in range(min(8, sample_images.size(0))):
        image = sample_images[i].cpu().permute(1, 2, 0).numpy()
        # 反归一化
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        # 创建可视化图
        fig, axes = plt.subplots(1, net.num_parts + 1, figsize=(4 * (net.num_parts + 1), 6))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 注意力图
        for j in range(net.num_parts):
            att = upsampled_attention[i, j].cpu().numpy()
            axes[j + 1].imshow(image)
            axes[j + 1].imshow(att, alpha=0.5, cmap='jet')
            axes[j + 1].set_title(f'Part {j+1}')
            axes[j + 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'best_attention_e{epoch}_sample{i}.png')
        plt.savefig(save_path)
        plt.close(fig)

# 绘图函数
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    
    # 保存曲线图到实验目录
    plot_path = os.path.join(exp_dir, "train.jpg")
    fig.savefig(plot_path)

# 主函数
def main():
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        
        # 更新学习率
        scheduler.step()
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # 定期保存检查点
        if epoch % 5 == 0 or current_lr < 1e-5:
            print(f"保存周期性检查点，epoch {epoch}")
            checkpoint = {
                'net_dict': net.state_dict(),
                'acc': 100. * (1 - test_err),
                'epoch': epoch,
            }
            epoch_ckpt_path = os.path.join(exp_checkpoint_dir, f"ckpt_epoch_{epoch}.t7")
            torch.save(checkpoint, epoch_ckpt_path)

if __name__ == '__main__':
    main()