import os
import shutil

mvtec_root = "D:/Projects/datasets/mvtec"
target_root = "D:/Projects/datasets/mvtec_test_all"

if not os.path.exists(target_root):
    os.makedirs(target_root)

for category in os.listdir(mvtec_root):
    train_dir = os.path.join(mvtec_root, category, "test", "good")  # 只取good图像用于训练
    if os.path.exists(train_dir):
        target_dir = os.path.join(target_root, category)
        os.makedirs(target_dir, exist_ok=True)
        for file in os.listdir(train_dir):
            shutil.copy(os.path.join(train_dir, file), os.path.join(target_dir, file))

print("训练集目录构建完成！")
