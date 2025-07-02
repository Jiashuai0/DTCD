import os
import shutil

mvtec_root = "D:/Projects/datasets/mvtec"
target_root = "D:/Projects/datasets/mvtec_test_all"

if not os.path.exists(target_root):
    os.makedirs(target_root)

# 定义缺陷类型处理顺序（按字母排序）
defect_order = ["broken_large", "broken_small", "contamination", "good"]

for category in os.listdir(mvtec_root):
    category_path = os.path.join(mvtec_root, category)
    test_dir = os.path.join(category_path, "test")
    target_category_dir = os.path.join(target_root, category)
    
    if not os.path.exists(test_dir) or not os.path.isdir(test_dir):
        continue
    
    os.makedirs(target_category_dir, exist_ok=True)
    
    # 获取目标目录当前文件数，确定起始编号
    existing_files = os.listdir(target_category_dir)
    start_idx = len(existing_files)  # 如已有20文件，则从020开始
    
    # 按顺序处理每个缺陷类型
    for defect_type in defect_order:
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.exists(defect_dir):
            continue
        
        # 遍历缺陷类型文件夹中的文件
        files = sorted(os.listdir(defect_dir))
        for idx, file in enumerate(files):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # 生成新文件名：起始编号 + 顺序号，补零至3位
            new_name = f"{(start_idx + idx):03d}{os.path.splitext(file)[1]}"
            src = os.path.join(defect_dir, file)
            dst = os.path.join(target_category_dir, new_name)
            
            shutil.copy(src, dst)
        
        # 更新起始编号（避免下一个缺陷类型覆盖）
        start_idx += len(files)

print("测试集已按顺序重命名完成！")