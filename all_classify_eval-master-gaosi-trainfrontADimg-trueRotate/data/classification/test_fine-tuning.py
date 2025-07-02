import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json

# === 参数配置 ===
num_classes = 15  # 根据实际类别数修改
batch_size = 32
model_path = "./logs/fine_tuning/fold_9_best.pth"  # 替换为实际模型路径
test_root = "/root/.vscode-server/data/Datasets/mvtec_classify/mvtec_test_all"  # 测试集路径
result_file = "./classification/classify_outputs/test_accuracy.json"  # 结果保存文件

# === 预处理（需与训练时一致） ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 加载测试集 ===
test_dataset = datasets.ImageFolder(test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === 加载模型 ===
model = models.resnet18(weights=None)  # 不加载预训练权重
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# === 评估函数 ===
def evaluate(model, dataloader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# === 运行评估 ===
accuracy = evaluate(model, test_loader)
print(f"测试集准确率: {accuracy:.4f}")

# === 保存结果 ===
with open(result_file, "w") as f:
    json.dump({"test_accuracy": accuracy}, f, indent=4)
print(f"结果已保存至 {result_file}")
