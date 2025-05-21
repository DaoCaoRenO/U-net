import os
import sys
# --- 新增：把项目根目录（u-net-voc2012）加入到模块搜索路径 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.unet import UNet
from torchvision.datasets import VOCSegmentation
from torch import nn, optim
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import random
import matplotlib
from matplotlib import colors

def setup_logger(log_dir):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志记录器
    logger = logging.getLogger('unet_training')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def visualize_results(images, masks, predictions, save_dir, epoch, filenames=None, indices=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 如果没有指定索引，随机选择几个样本
    if indices is None:
        indices = random.sample(range(len(images)), min(5, len(images)))
    
    for i, idx in enumerate(indices):
        img = images[idx].cpu()
        mask = masks[idx].cpu()
        pred = predictions[idx].cpu()
        
        # 反标准化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
        img = np.clip(img, 0, 1)
        # 21类，0为背景
        cmap = plt.get_cmap('tab20', 21)
        newcolors = cmap(np.arange(21))
        newcolors[0] = [0, 0, 0, 1]  # 背景设为黑色
        custom_cmap = colors.ListedColormap(newcolors)
        # 创建图像对比
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask.numpy(), cmap=custom_cmap, vmin=0, vmax=20)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
      
        axes[2].imshow(pred.numpy(), cmap=custom_cmap, vmin=0, vmax=20)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        if filenames is not None:
            prefix = os.path.splitext(os.path.basename(filenames[idx]))[0]
            save_name = f"epoch{epoch}_{prefix}_sample{i}.png"
        else:
            save_name = f"epoch_{epoch}_sample_{i}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.close()

def calculate_iou(pred, target, n_classes):
    """
    计算类别IoU和平均IoU
    pred: (B, H, W) 预测类别索引
    target: (B, H, W) 真实类别索引
    n_classes: 类别数量
    
    返回: 各类IoU字典和平均IoU
    """
    ious = {}
    # 忽略255标签(未标注)
    mask = (target != 255)
    
    for cls in range(n_classes):
        # 创建当前类的掩码
        pred_inds = pred == cls
        target_inds = target == cls
        
        # 仅在有效区域内计算
        pred_inds = pred_inds & mask
        target_inds = target_inds & mask
        
        # 计算交集和并集
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        # 避免除零错误
        iou = intersection / union if union > 0 else 0.0
        ious[cls] = iou
    
    # 计算平均IoU (mIoU)
    valid_classes = [cls for cls in range(n_classes) if 
                    (pred == cls).sum().item() > 0 or (target == cls).sum().item() > 0]
    if valid_classes:
        mean_iou = sum(ious[cls] for cls in valid_classes) / len(valid_classes)
    else:
        mean_iou = 0.0
    
    return ious, mean_iou

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, logger, save_dir,config):
    best_val_loss = float('inf')
    best_val_miou = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        train_ious = []
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 计算像素准确率和IoU
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == masks).sum().item()
            total = masks.numel() - (masks == 255).sum().item()  # 忽略255标签
            accuracy = correct / total if total > 0 else 0
            running_acc += accuracy
            
            # 计算当前批次的IoU
            _, batch_miou = calculate_iou(pred.cpu(), masks.cpu(), config['num_classes'])
            train_ious.append(batch_miou)
            
            # 记录训练进度
            if (i+1) % 1 == 0:  # 每10个批次记录一次
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, mIoU: {batch_miou:.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        train_miou = sum(train_ious) / len(train_ious) if train_ious else 0.0
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, '
                  f'Accuracy: {train_acc:.4f}, mIoU: {train_miou:.4f}')
        
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_ious = []
        class_ious = {cls: [] for cls in range(config['num_classes'])}
        
        # 初始化可视化变量到循环外部
        val_images = []
        val_masks = []
        val_preds = []
        val_filenames = []
        
        with torch.no_grad():
            for b, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs, dim=1)
                correct = (pred == masks).sum().item()
                total = masks.numel() - (masks == 255).sum().item()
                accuracy = correct / total if total > 0 else 0
                val_acc += accuracy
                
                cls_ious, batch_miou = calculate_iou(pred.cpu(), masks.cpu(), config['num_classes'])
                val_ious.append(batch_miou)
                
                for cls in range(config['num_classes']):
                    if cls in cls_ious:
                        class_ious[cls].append(cls_ious[cls])
                
                # 仅从前几个批次收集可视化样本，注意这里不再每次迭代都重置数组
                if b < 2 and len(val_images) < 10:  # 只从前2个批次收集最多10个样本
                    for j in range(min(5, len(images))):
                        val_images.append(images[j].cpu())  # 确保CPU版本
                        val_masks.append(masks[j].cpu())
                        val_preds.append(pred[j].cpu())
                        
                        # 获取VOC文件名
                        img_idx = b * val_loader.batch_size + j
                        if hasattr(val_loader.dataset, 'images') and img_idx < len(val_loader.dataset.images):
                            img_path = val_loader.dataset.images[img_idx]
                            val_filenames.append(img_path)
                        else:
                            val_filenames.append(f'sample_{img_idx}')
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        val_miou = sum(val_ious) / len(val_ious) if val_ious else 0.0
        
        # 计算每个类别的平均IoU
        avg_class_ious = {cls: sum(ious)/len(ious) if ious else 0.0 
                          for cls, ious in class_ious.items()}
        
        # 记录详细的IoU信息
        iou_details = ', '.join([f'Class {cls}: {iou:.4f}' for cls, iou in avg_class_ious.items() 
                               if sum(class_ious[cls]) > 0])
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, '
                  f'Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}')
        logger.info(f'类别IoUs: {iou_details}')
        
        # 随机选择验证样本进行可视化
        indices = random.sample(range(len(val_images)), min(5, len(val_images)))
        # 调用可视化时传递文件名
       # 可视化验证样本
        if val_images:  # 确保有图像再尝试可视化
            logger.info(f"收集了 {len(val_images)} 张图像用于可视化")
            # 随机选择验证样本进行可视化
            indices = random.sample(range(len(val_images)), min(5, len(val_images)))
            
            vis_dir = os.path.join(save_dir, 'visualizations')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
                
            visualize_results(
                val_images,
                val_masks,
                val_preds,
                vis_dir,
                epoch + 1,
                filenames=val_filenames,
                indices=indices
            )
            logger.info(f"已保存可视化结果到 {vis_dir}")
        
        # 保存最佳mIoU模型
        # if val_miou > best_val_miou:
        #     best_val_miou = val_miou
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'val_acc': val_acc,
        #         'val_miou': val_miou,
        #         'device': str(device)
        #     }, os.path.join(save_dir, 'best_miou_model.pth'))
        #     logger.info(f'最佳mIoU模型已保存 (验证mIoU: {val_miou:.4f})')
        
        # # 保存最佳模型(基于验证损失)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'val_acc': val_acc,
        #         'val_miou': val_miou,
        #         'device': str(device)
        #     }, os.path.join(save_dir, 'best_loss_model.pth'))
        #     logger.info(f'最佳损失模型已保存 (验证损失: {val_loss:.4f})')
        
        # # 保存每个epoch的模型
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'val_loss': val_loss,
        #     'val_acc': val_acc,
        #     'val_miou': val_miou,
        #     'device': str(device)
        # }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))

def main():
    config = load_config('../configs/config.yaml')
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('../outputs', f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(save_dir)
    
    # 检测是否有可用的GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 图像变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # mask 变换
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
    ])
    
    train_dataset = VOCSegmentation(
        root='../../../DeepLearn2/data',
        year='2012',
        image_set='train',
        download=False,
        transform=train_transform,
        target_transform=mask_transform
    )
    val_dataset = VOCSegmentation(
        root='../../../DeepLearn2/data',
        year='2012',
        image_set='val',
        download=False,
        transform=train_transform,
        target_transform=mask_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = UNet(in_channels=3, n_classes=config['num_classes'])
    model.to(device)
    
    # 设置类别权重：假设背景类别（索引0）权重设低，其它类别权重为1
    weights = torch.ones(config['num_classes'])
    weights[0] = 0.1  # 背景类别权重较低
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练和验证模型
    train_model(model, train_loader, val_loader, criterion, optimizer,               
               config['num_epochs'], device, logger, save_dir, config)
    
    logger.info('训练完成！')

if __name__ == '__main__':
    main()