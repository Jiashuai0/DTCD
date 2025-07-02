import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import os

def main():
    # ================= 硬件配置 =================
    device_id = 3  # 与--gpu_id参数对应
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"\n=== 正在使用GPU: {torch.cuda.get_device_name(device_id)} ===")
    else:
        print("\n=== 警告: 使用CPU进行训练，速度可能较慢 ===")
    
    # ================= 参数配置 =================
    config = {
        "num_classes": 15,          # 实际类别数
        "batch_size": 32,           # 批大小
        "learning_rate": 0.001,     # 初始学习率
        "num_epochs": 10,           # 总训练轮数
        "num_folds": 10,            # K折交叉验证数
        "dataset_root": "/root/.vscode-server/data/Datasets/mvtec_classify/mvtec_train_all",
        "log_dir": "./logs/fine_tuning",  # 日志保存路径
        "momentum": 0.9,            # 优化器动量
        "weight_decay": 1e-4,       # 权重衰减
        "num_workers": 4            # 数据加载线程数
    }

    # ================= 环境验证 =================
    print("\n=== 环境验证 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"总显存: {torch.cuda.get_device_properties(device_id).total_memory/1024**3:.1f}GB")

    # ================= 数据准备 =================
    try:
        full_dataset, labels = load_dataset(config["dataset_root"])
        print(f"\n=== 数据集加载成功 ===")
        print(f"样本总数: {len(full_dataset)}")
        print(f"类别数: {len(full_dataset.classes)}")
    except Exception as e:
        print(f"\n!!! 数据集加载失败: {str(e)}")
        return

    # ================= 交叉验证训练 =================
    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset, labels)):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}/{config['num_folds']} 训练开始")
        print(f"{'='*40}")

        # 初始化模型和训练组件
        model = initialize_model(config["num_classes"], device_id)
        optimizer = configure_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], f"fold_{fold}"))

        # 准备数据加载器
        train_loader, val_loader = create_dataloaders(
            full_dataset, train_idx, val_idx, 
            config["batch_size"], config["num_workers"]
        )

        # 执行训练流程
        best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            writer=writer,
            fold=fold
        )
        
        print(f"\nFold {fold+1} 最佳模型已保存至: {best_model_path}")
        writer.close()

def load_dataset(dataset_root: str):
    """加载并预处理数据集"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_root}")
    
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    labels = [label for _, label in dataset]
    return dataset, labels

def initialize_model(num_classes: int, device_id: int) -> nn.Module:
    """初始化预训练模型"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 冻结除最后两个层外的参数
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True
    
    return model.to(device)

def configure_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """配置优化器"""
    trainable_params = [
        {"params": model.layer4.parameters()},
        {"params": model.fc.parameters()}
    ]
    
    return optim.SGD(
        trainable_params,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

def create_dataloaders(dataset, train_idx, val_idx, batch_size: int, num_workers: int):
    """创建训练和验证数据加载器"""
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, criterion, config, writer, fold: int):
    """执行完整的训练流程"""
    device = next(model.parameters()).device  # 获取模型所在的设备
    scaler = torch.cuda.amp.GradScaler()
    best_val_acc = 0.0
    best_model_path = ""
    
    for epoch in range(config["num_epochs"]):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device
        )
        
        # 验证阶段
        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # 记录指标
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)
        
        # 打印训练进度
        print(f"[Fold {fold+1} Epoch {epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(config["log_dir"], f"fold_{fold}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ 发现Fold {fold+1}最佳模型 | 验证准确率: {val_acc:.2%}")
    
    return best_model_path

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for images, labels in loader:
        # 数据转移至设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计指标
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / total_samples, correct / total_samples

def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss / total_samples, correct / total_samples

if __name__ == "__main__":
    # 设置环境变量（可选）
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 调试用，可显示具体CUDA错误
    main()