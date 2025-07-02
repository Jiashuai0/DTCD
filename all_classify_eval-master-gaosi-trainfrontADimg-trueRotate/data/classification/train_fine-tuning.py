import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import os
# === 新增：强制设置CUDA设备 ===
torch.cuda.set_device(3)  # 显式指定使用3号GPU（与--gpu_id 3参数对应）

# === 参数 ===
num_classes = 15  # 修改为实际类别数
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_folds = 10
dataset_root = "/root/.vscode-server/data/Datasets/mvtec_classify/mvtec_train_all"
log_base_dir = "./logs"

# === 新增：CUDA可用性验证 ===
print("\n=== CUDA环境验证 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"当前GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB\n")

# === 预处理 ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 加载数据集 ===
full_dataset = datasets.ImageFolder(dataset_root, transform=transform)
labels = [label for _, label in full_dataset]  # 用于分层抽样

# === Stratified K-Fold ===
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset, labels)):
    print(f"\n Fold {fold+1}/{num_folds}")
    
    # 路径设置
    fold_log_dir = os.path.join(log_base_dir, f"tensorboard/fold_{fold}")
    fold_model_path = os.path.join(log_base_dir, f"fold_{fold}_best_model.pth")
    writer = SummaryWriter(log_dir=fold_log_dir)

    # 子集划分
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 冻结除 layer4 和 fc 外的所有层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # 自动混合精度训练

    # 优化器
    optimizer = optim.SGD(
        list(model.layer4.parameters()) + list(model.fc.parameters()),
        lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)  # 异步传输
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 自动混合精度
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        # 验证
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss /= total
        val_acc = correct / total

        # 日志记录
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)

        print(f"[Fold {fold+1} Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), fold_model_path)
            print(f"Saved best model for Fold {fold+1} with val acc: {val_acc:.4f}")

    writer.close()

# === 入口判断 ===
if __name__ == "__main__":
    main()