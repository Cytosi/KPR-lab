"""
MNIST手写数字识别实验
实验内容：
1. 数据准备：加载MNIST数据集，数据预处理，展示样本
2. 模型设计：实现基础CNN(模型A)和改进CNN(模型B)
3. 模型训练：使用Adam优化器，交叉熵损失，记录训练曲线
4. 模型评估：混淆矩阵，指标计算，错误样本展示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

os.makedirs('outputs', exist_ok=True)


# ==================== 一、数据准备 ====================

def compute_dataset_statistics():
    """
    (1) 计算MNIST训练集的数据分布：均值和方差
    """
    print("\n" + "="*60)
    print("一、数据准备")
    print("="*60)
    print("\n(1) 计算MNIST训练集的数据分布")
    print("-" * 40)
    
    # 加载原始数据（不进行归一化）
    raw_transform = transforms.ToTensor()
    raw_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=raw_transform)
    
    # 计算均值和方差
    all_pixels = []
    for img, _ in raw_dataset:
        all_pixels.append(img.numpy().flatten())
    
    all_pixels = np.concatenate(all_pixels)
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    var = np.var(all_pixels)
    
    print(f"训练集样本数: {len(raw_dataset)}")
    print(f"图像尺寸: 28 x 28 x 1 (灰度图)")
    print(f"像素值范围: [0, 1] (ToTensor自动归一化)")
    print()
    print(f">>> 均值 (Mean): {mean:.6f}")
    print(f">>> 方差 (Variance): {var:.6f}")
    print(f">>> 标准差 (Std): {std:.6f}")
    print()
    print("代码片段:")
    print("```python")
    print("# 计算MNIST训练集的均值和方差")
    print("raw_transform = transforms.ToTensor()")
    print("raw_dataset = datasets.MNIST(root='./data', train=True, transform=raw_transform)")
    print("all_pixels = np.concatenate([img.numpy().flatten() for img, _ in raw_dataset])")
    print("mean = np.mean(all_pixels)  # 均值")
    print("std = np.std(all_pixels)    # 标准差")
    print("var = np.var(all_pixels)    # 方差")
    print("```")
    
    return mean, std


def get_data_loaders(batch_size=64):
    """加载MNIST数据集并进行预处理"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n数据加载完成:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def show_samples(dataset, samples_per_class=5):
    """
    (2) 展示处理后的样本（每个类别随机展示至少5个样本）
    """
    print("\n(2) 展示处理后的图像样本")
    print("-" * 40)
    
    fig, axes = plt.subplots(10, samples_per_class, figsize=(samples_per_class*2, 20))
    fig.suptitle('MNIST Normalized Samples (5 samples per class)', fontsize=16, fontweight='bold')
    
    class_samples = {i: [] for i in range(10)}
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    for idx in indices:
        img, label = dataset[idx]
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break
    
    for class_idx in range(10):
        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]
            img = class_samples[class_idx][sample_idx].squeeze()
            # 反归一化用于显示
            img = img * 0.3081 + 0.1307
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if sample_idx == 0:
                ax.set_ylabel(f'Digit {class_idx}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/1_sample_display.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("样本展示图已保存: outputs/1_sample_display.png")


# ==================== 二、模型设计 ====================

class ModelA(nn.Module):
    """
    模型A - 基础CNN
    
    架构说明:
    - 2个卷积层 (Conv2d)
    - 最大池化层 (MaxPool2d)
    - 2个全连接层 (Linear)
    - ReLU激活函数
    
    Conv2d参数含义:
    - in_channels: 输入通道数 (灰度图为1, RGB为3)
    - out_channels: 输出通道数/卷积核数量
    - kernel_size: 卷积核大小
    - padding: 填充大小, 保持特征图尺寸
    """
    def __init__(self):
        super(ModelA, self).__init__()
        # Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 输入: 1通道(灰度图), 输出: 32个特征图, 3x3卷积核, padding=1保持尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 输入: 32通道, 输出: 64个特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # MaxPool2d: 2x2池化, 步长2, 特征图尺寸减半
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层1: 64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 全连接层2(输出层): 128 -> 10 (10个类别)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 输入: [N, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # -> [N, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [N, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)            # 展平 -> [N, 3136]
        x = F.relu(self.fc1(x))               # -> [N, 128]
        x = self.fc2(x)                       # -> [N, 10]
        return x


class ModelB(nn.Module):
    """
    模型B - 改进CNN
    
    优化架构:
    1. 增加网络深度: 4个卷积层 (对比模型A的2个)
    2. BatchNorm层: 批量归一化, 加速训练, 提高稳定性
    3. Dropout: 随机丢弃神经元, 防止过拟合
    4. 全局平均池化: 替代全连接层, 减少参数量
    5. Leaky ReLU: 解决ReLU的"死亡神经元"问题
    
    BatchNorm作用:
    - 对每个batch的数据进行归一化
    - 加速收敛, 允许使用更大学习率
    - 有轻微正则化效果
    
    Dropout作用:
    - 训练时随机丢弃部分神经元
    - 减少神经元之间的共适应
    - 有效防止过拟合
    """
    def __init__(self, dropout_rate=0.25):
        super(ModelB, self).__init__()
        
        # 第一个卷积块: 2个卷积层
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        # 第二个卷积块: 2个卷积层
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        
        # 全局平均池化: 将特征图压缩为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)      # [N, 1, 28, 28] -> [N, 32, 14, 14]
        x = self.conv_block2(x)      # -> [N, 64, 7, 7]
        x = self.conv_block3(x)      # -> [N, 128, 7, 7]
        x = self.global_avg_pool(x)  # -> [N, 128, 1, 1]
        x = x.view(x.size(0), -1)    # -> [N, 128]
        x = self.fc(x)               # -> [N, 10]
        return x


def print_model_architecture():
    """(1)(2) 打印模型架构和参数量"""
    print("\n" + "="*60)
    print("二、模型设计")
    print("="*60)
    
    model_a = ModelA()
    model_b = ModelB()
    
    # 模型A
    print("\n(1) 模型A架构 - 基础CNN")
    print("-" * 40)
    print(model_a)
    
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"\n>>> 模型A总参数量: {params_a:,}")
    
    print("\n参数量计算:")
    print("  conv1: 1×32×3×3 + 32 = 320")
    print("  conv2: 32×64×3×3 + 64 = 18,496")
    print("  fc1: 64×7×7×128 + 128 = 401,536")
    print("  fc2: 128×10 + 10 = 1,290")
    print(f"  总计: {params_a:,}")
    
    print("\nConv2d参数含义:")
    print("  - in_channels: 输入通道数 (1=灰度图)")
    print("  - out_channels: 输出通道数/卷积核数量")
    print("  - kernel_size: 卷积核大小 (3表示3×3)")
    print("  - padding: 边缘填充 (1保持特征图尺寸)")
    
    # 模型B
    print("\n" + "-" * 40)
    print("(2) 模型B架构 - 改进CNN")
    print("-" * 40)
    print(model_b)
    
    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"\n>>> 模型B总参数量: {params_b:,}")
    
    print("\n模型B优化点:")
    print("  1. 4个卷积层 (增加深度)")
    print("  2. BatchNorm层 (加速训练)")
    print("  3. Dropout (防止过拟合)")
    print("  4. 全局平均池化 (减少参数)")
    print("  5. Leaky ReLU (避免死亡神经元)")
    
    # (3) BatchNorm和Dropout解释
    print("\n" + "-" * 40)
    print("(3) BatchNorm和Dropout的作用")
    print("-" * 40)
    print("\nBatchNorm (批量归一化):")
    print("  - 对每个mini-batch的特征进行归一化")
    print("  - 使得每层输入分布稳定")
    print("  - 加速模型收敛, 可使用更大学习率")
    print("  - 有轻微正则化效果")
    
    print("\nDropout (随机丢弃):")
    print("  - 训练时以概率p随机将神经元输出置0")
    print("  - 减少神经元间的共适应性")
    print("  - 等效于训练多个子网络的集成")
    print("  - 有效防止过拟合")
    
    return params_a, params_b


# ==================== 三、模型训练 ====================

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model'):
    """训练模型并记录详细指标"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    train_accs = []
    test_accs = []
    steps = []
    epoch_times = []
    epoch_train_accs = []
    epoch_test_accs = []
    step = 0
    
    print(f"\n{'='*50}")
    print(f"开始训练 {model_name}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            step += 1
            
            # 每50个batch记录一次
            if batch_idx % 50 == 0:
                train_losses.append(loss.item())
                train_accs.append(100. * correct / total)
                steps.append(step)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        train_acc = 100. * correct / total
        test_acc = evaluate_accuracy(model, test_loader)
        epoch_train_accs.append(train_acc)
        epoch_test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'Loss: {epoch_loss/len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'steps': steps,
        'epoch_times': epoch_times,
        'epoch_train_accs': epoch_train_accs,
        'epoch_test_accs': epoch_test_accs
    }


def evaluate_accuracy(model, data_loader):
    """计算准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return 100. * correct / total


def plot_training_curves_separate(results_a, results_b):
    """
    (1)(2) 分别绘制模型A和模型B的训练曲线
    """
    print("\n" + "="*60)
    print("三、模型训练")
    print("="*60)
    
    # 模型A曲线
    print("\n(1) 模型A训练曲线")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model A Training Curves', fontsize=14, fontweight='bold')
    
    axes[0].plot(results_a['steps'], results_a['train_losses'], 'b-', alpha=0.8)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_a['steps'], results_a['train_accs'], 'b-', alpha=0.8)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/3_model_a_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("模型A曲线已保存: outputs/3_model_a_curves.png")
    
    print("\n模型A各Epoch训练时间:")
    for i, t in enumerate(results_a['epoch_times']):
        print(f"  Epoch {i+1}: {t:.2f}s")
    print(f"  总训练时间: {sum(results_a['epoch_times']):.2f}s")
    
    # 模型B曲线
    print("\n(2) 模型B训练曲线")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model B Training Curves', fontsize=14, fontweight='bold')
    
    axes[0].plot(results_b['steps'], results_b['train_losses'], 'r-', alpha=0.8)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_b['steps'], results_b['train_accs'], 'r-', alpha=0.8)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/3_model_b_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("模型B曲线已保存: outputs/3_model_b_curves.png")
    
    print("\n模型B各Epoch训练时间:")
    for i, t in enumerate(results_b['epoch_times']):
        print(f"  Epoch {i+1}: {t:.2f}s")
    print(f"  总训练时间: {sum(results_b['epoch_times']):.2f}s")
    
    # 对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model A vs Model B Training Comparison', fontsize=14, fontweight='bold')
    
    axes[0].plot(results_a['steps'], results_a['train_losses'], 'b-', label='Model A', alpha=0.8)
    axes[0].plot(results_b['steps'], results_b['train_losses'], 'r-', label='Model B', alpha=0.8)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_a['steps'], results_a['train_accs'], 'b-', label='Model A', alpha=0.8)
    axes[1].plot(results_b['steps'], results_b['train_accs'], 'r-', label='Model B', alpha=0.8)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/3_comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n对比曲线已保存: outputs/3_comparison_curves.png")
    
    # (3) 解释准确率和损失曲线不完全正比
    print("\n(3) 准确率曲线和损失曲线为什么不完全正比?")
    print("-" * 40)
    print("""
原因分析:
1. 损失函数的非线性: 交叉熵损失是概率的对数, 当模型越来越自信
   (预测概率接近1), 损失下降速度会变慢, 但准确率可能已经很高

2. 阈值效应: 准确率是离散的(预测对或错), 而损失是连续的
   - 例如: 将预测概率从0.6提升到0.9, 准确率不变(都算对)
   - 但损失会显著下降

3. 边界样本: 模型可能在某些"难样本"上反复调整
   - 准确率在0.5附近波动
   - 损失持续下降(概率在改善)

4. 过拟合影响: 模型可能开始记忆训练样本
   - 训练准确率继续上升
   - 但对某些样本过度自信, 错误样本的损失增加
""")


# ==================== 四、模型评估 ====================

def get_predictions(model, data_loader):
    """获取模型预测结果"""
    model.eval()
    all_preds = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_images.extend(data.cpu())
    
    return np.array(all_preds), np.array(all_targets), all_images


def evaluate_model(model, test_loader, model_name, results):
    """完整评估模型"""
    preds, targets, images = get_predictions(model, test_loader)
    
    # 混淆矩阵
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'outputs/4_{model_name.lower().replace(" ", "_")}_confusion.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {filename}")
    
    # 计算指标
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    f1 = f1_score(targets, preds, average='weighted')
    
    print(f"\n评估指标:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # 错误分类样本
    misclassified_idx = np.where(targets != preds)[0]
    num_errors = len(misclassified_idx)
    print(f"\n错误分类样本数: {num_errors}")
    
    if num_errors > 0:
        num_to_show = min(5, num_errors)
        selected_idx = np.random.choice(misclassified_idx, num_to_show, replace=False)
        
        fig, axes = plt.subplots(1, num_to_show, figsize=(num_to_show * 3, 4))
        if num_to_show == 1:
            axes = [axes]
        
        fig.suptitle(f'{model_name} - Misclassified Samples', fontsize=14, fontweight='bold')
        
        for i, idx in enumerate(selected_idx):
            img = images[idx].squeeze()
            img = img * 0.3081 + 0.1307
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {targets[idx]}\nPred: {preds[idx]}', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        filename = f'outputs/4_{model_name.lower().replace(" ", "_")}_errors.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"错误样本已保存: {filename}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_errors': num_errors,
        'train_accs': results['epoch_train_accs'],
        'test_accs': results['epoch_test_accs']
    }


def analyze_models(metrics_a, metrics_b, results_a, results_b):
    """(3)(4) 分析模型训练效率和过拟合"""
    print("\n(3) 训练效率分析")
    print("-" * 40)
    
    # 计算相同时间内的准确率提升
    time_a = sum(results_a['epoch_times'])
    time_b = sum(results_b['epoch_times'])
    
    acc_improve_a = results_a['epoch_test_accs'][-1] - results_a['epoch_test_accs'][0]
    acc_improve_b = results_b['epoch_test_accs'][-1] - results_b['epoch_test_accs'][0]
    
    efficiency_a = acc_improve_a / time_a
    efficiency_b = acc_improve_b / time_b
    
    print(f"模型A:")
    print(f"  总训练时间: {time_a:.2f}s")
    print(f"  测试集准确率提升: {acc_improve_a:.2f}%")
    print(f"  训练效率: {efficiency_a:.4f}%/s")
    
    print(f"\n模型B:")
    print(f"  总训练时间: {time_b:.2f}s")
    print(f"  测试集准确率提升: {acc_improve_b:.2f}%")
    print(f"  训练效率: {efficiency_b:.4f}%/s")
    
    if efficiency_a > efficiency_b:
        print(f"\n>>> 结论: 模型A训练效率更高 ({efficiency_a:.4f} > {efficiency_b:.4f})")
    else:
        print(f"\n>>> 结论: 模型B训练效率更高 ({efficiency_b:.4f} > {efficiency_a:.4f})")
    
    print("\n(4) 过拟合分析")
    print("-" * 40)
    
    # 计算训练集和测试集准确率差异
    gap_a = results_a['epoch_train_accs'][-1] - results_a['epoch_test_accs'][-1]
    gap_b = results_b['epoch_train_accs'][-1] - results_b['epoch_test_accs'][-1]
    
    print(f"模型A:")
    print(f"  最终训练集准确率: {results_a['epoch_train_accs'][-1]:.2f}%")
    print(f"  最终测试集准确率: {results_a['epoch_test_accs'][-1]:.2f}%")
    print(f"  准确率差距 (Gap): {gap_a:.2f}%")
    
    print(f"\n模型B:")
    print(f"  最终训练集准确率: {results_b['epoch_train_accs'][-1]:.2f}%")
    print(f"  最终测试集准确率: {results_b['epoch_test_accs'][-1]:.2f}%")
    print(f"  准确率差距 (Gap): {gap_b:.2f}%")
    
    print("\n过拟合判断标准: 训练集准确率远高于测试集准确率")
    
    if gap_a > gap_b:
        print(f"\n>>> 结论: 模型A更容易过拟合 (Gap: {gap_a:.2f}% > {gap_b:.2f}%)")
        print("    原因: 模型A没有Dropout和BatchNorm等正则化技术")
    else:
        print(f"\n>>> 结论: 模型B更容易过拟合 (Gap: {gap_b:.2f}% > {gap_a:.2f}%)")
    
    # 绘制训练/测试准确率对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Train vs Test Accuracy (Overfitting Analysis)', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(results_a['epoch_train_accs']) + 1)
    
    axes[0].plot(epochs, results_a['epoch_train_accs'], 'b-', label='Train Acc', marker='o')
    axes[0].plot(epochs, results_a['epoch_test_accs'], 'b--', label='Test Acc', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Model A (Gap: {gap_a:.2f}%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, results_b['epoch_train_accs'], 'r-', label='Train Acc', marker='o')
    axes[1].plot(epochs, results_b['epoch_test_accs'], 'r--', label='Test Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Model B (Gap: {gap_b:.2f}%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/4_overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n过拟合分析图已保存: outputs/4_overfitting_analysis.png")


# ==================== 主程序 ====================

def main():
    print("="*60)
    print("MNIST手写数字识别实验报告")
    print("="*60)
    
    # 一、数据准备
    mean, std = compute_dataset_statistics()
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=64)
    show_samples(train_dataset, samples_per_class=5)
    
    # 二、模型设计
    params_a, params_b = print_model_architecture()
    
    # 三、模型训练
    model_a = ModelA()
    model_b = ModelB(dropout_rate=0.25)
    
    results_a = train_model(model_a, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model A')
    results_b = train_model(model_b, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model B')
    
    plot_training_curves_separate(results_a, results_b)
    
    # 四、模型评估
    print("\n" + "="*60)
    print("四、模型评估")
    print("="*60)
    
    print("\n(1) 模型A评估")
    print("-" * 40)
    metrics_a = evaluate_model(results_a['model'], test_loader, 'Model A', results_a)
    
    print("\n(2) 模型B评估")
    print("-" * 40)
    metrics_b = evaluate_model(results_b['model'], test_loader, 'Model B', results_b)
    
    analyze_models(metrics_a, metrics_b, results_a, results_b)
    
    # 结果总结
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    print(f"\nModel A (Basic CNN):")
    print(f"  - 参数量: {params_a:,}")
    print(f"  - 测试集准确率: {metrics_a['accuracy']*100:.2f}%")
    
    print(f"\nModel B (Improved CNN):")
    print(f"  - 参数量: {params_b:,}")
    print(f"  - 测试集准确率: {metrics_b['accuracy']*100:.2f}%")
    
    if metrics_b['accuracy'] >= 0.99:
        print(f"\n✓ Model B 达到99%以上准确率要求!")
    else:
        print(f"\n✗ Model B 未达到99%准确率要求，当前: {metrics_b['accuracy']*100:.2f}%")
    
    print("\n生成的输出文件:")
    print("  [一] outputs/1_sample_display.png - 样本展示")
    print("  [三] outputs/3_model_a_curves.png - 模型A训练曲线")
    print("  [三] outputs/3_model_b_curves.png - 模型B训练曲线")
    print("  [三] outputs/3_comparison_curves.png - 对比曲线")
    print("  [四] outputs/4_model_a_confusion.png - 模型A混淆矩阵")
    print("  [四] outputs/4_model_b_confusion.png - 模型B混淆矩阵")
    print("  [四] outputs/4_model_a_errors.png - 模型A错误样本")
    print("  [四] outputs/4_model_b_errors.png - 模型B错误样本")
    print("  [四] outputs/4_overfitting_analysis.png - 过拟合分析")
    
    # 保存模型
    torch.save(results_a['model'].state_dict(), 'outputs/model_a.pth')
    torch.save(results_b['model'].state_dict(), 'outputs/model_b.pth')
    print("\n模型权重已保存到 outputs/ 目录")


if __name__ == '__main__':
    main()
