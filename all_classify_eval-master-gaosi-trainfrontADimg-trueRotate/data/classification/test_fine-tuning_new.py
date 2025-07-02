import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # ================= 硬件配置 =================
    device_id = 3  # 与训练时使用的GPU一致
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # ================= 参数配置 =================
    config = {
        "num_classes": 15,          # 必须与训练时一致
        "batch_size": 32,            # 与训练时一致
        "model_path": "./logs/fine_tuning/fold_9_best.pth",
        "test_root": "/root/.vscode-server/data/Datasets/mvtec_classify/mvtec_test_all",
        "result_file": "./classification/classify_outputs/test_accuracy.json"
    }

    # ================= 环境验证 =================
    print("\n=== 测试环境验证 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"当前使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU内存总量: {torch.cuda.get_device_properties(device).total_memory/1024**3:.1f}GB")

    # ================= 数据加载 =================
    try:
        # 预处理（必须与训练时一致）
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(config["test_root"], transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"\n=== 测试集加载成功 ===")
        print(f"测试样本总数: {len(test_dataset)}")
        print(f"类别列表: {test_dataset.classes}")
    except Exception as e:
        print(f"\n!!! 测试集加载失败: {str(e)}")
        return

    # ================= 模型加载 =================
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
        
        # 加载训练好的权重
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        model = model.to(device)
        model.eval()
        print("\n=== 模型加载成功 ===")
        print(f"模型结构: {model.__class__.__name__}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"\n!!! 模型加载失败: {str(e)}")
        return

    # ================= 执行测试 =================
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return correct / total

    print("\n=== 开始测试 ===")
    accuracy = evaluate_model(model, test_loader)
    print(f"测试准确率: {accuracy:.4f}")

    # ================= 保存结果 =================
    os.makedirs(os.path.dirname(config["result_file"]), exist_ok=True)
    with open(config["result_file"], "w") as f:
        json.dump({
            "model_path": config["model_path"],
            "test_samples": len(test_dataset),
            "accuracy": accuracy,
            "class_mapping": {i: cls for i, cls in enumerate(test_dataset.classes)}
        }, f, indent=4)
    
    print(f"\n=== 结果已保存至: {config['result_file']} ===")

if __name__ == "__main__":
    # 设置环境变量（可选）
    os.environ["CUDA_LAUNCH_BLOCKING"] = "2"  # 调试用
    main()