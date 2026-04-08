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




def compute_dataset_statistics():
    # 加载原始数据
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
    print()
    print(f">>> 均值 (Mean): {mean:.6f}")
    print(f">>> 方差 (Variance): {var:.6f}")
    print(f">>> 标准差 (Std): {std:.6f}")
    print()  
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




class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
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
    model_a = ModelA()
    model_b = ModelB()
    

    print(model_a)
    print(model_b)
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"\n>>> 模型A总参数量: {params_a:,}")       
    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"\n>>> 模型B总参数量: {params_b:,}")
    return params_a, params_b



def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model'):
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



def main():
    
    mean, std = compute_dataset_statistics()
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=64)
    show_samples(train_dataset, samples_per_class=5)
    
    params_a, params_b = print_model_architecture()
    
    model_a = ModelA()
    model_b = ModelB(dropout_rate=0.25)
    
    results_a = train_model(model_a, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model A')
    results_b = train_model(model_b, train_loader, test_loader, epochs=10, lr=0.001, model_name='Model B')
    
    plot_training_curves_separate(results_a, results_b)
    
    print("-" * 40)
    metrics_a = evaluate_model(results_a['model'], test_loader, 'Model A', results_a)
    
    print("-" * 40)
    metrics_b = evaluate_model(results_b['model'], test_loader, 'Model B', results_b)
    
    analyze_models(metrics_a, metrics_b, results_a, results_b)
    
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
    
    # 保存模型
    torch.save(results_a['model'].state_dict(), 'outputs/model_a.pth')
    torch.save(results_b['model'].state_dict(), 'outputs/model_b.pth')
    print("\n模型权重已保存到 outputs/ 目录")


if __name__ == '__main__':
    main()
