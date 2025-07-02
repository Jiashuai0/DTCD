import argparse
import os
import shutil
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision
import numpy as np
import cv2
from PIL import Image
import csv

# 从constant.py导入ALL_CATEGORY
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.destseg_old import EnhancedDeSTSeg, FineTunedClassifier
from model.metrics import AUPRO, IAPS
warnings.filterwarnings("ignore")

# 指定需要处理的类别 - 注意使用下划线代替空格
SPECIAL_CATEGORIES = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
                      'pill', 'screw', 'toothbrush', 'transistor']

def create_colormap(anomaly_map):
    """将异常图转换为彩色热力图"""
    # 确保异常图在0-1范围内
    anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map) + 1e-8)
    
    # 转换为0-255范围
    anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
    
    # 应用JET颜色映射
    colormap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    
    return colormap

def evaluate(args, category, model, visualizer, global_step=0):
    model.eval()

    # 加载分类器并设置为评估模式
    classifier = FineTunedClassifier().cuda()
    classifier_path = "/root/.vscode-server/data/all_classify_eval-master/logs/fine_tuning/fold_9_best.pth"
    state_dict = torch.load(classifier_path)
    
    # 调整键前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f'base_model.{key}'
        new_state_dict[new_key] = value
    
    classifier.load_state_dict(new_state_dict, strict=True)
    classifier.eval()

    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=os.path.join(args.mvtec_path, category, "test"),
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        
        # 初始化指标
        de_st_IAPS = IAPS().cuda()
        de_st_AUPRO = AUPRO().cuda()
        de_st_AUROC = AUROC(task='binary').cuda()
        de_st_AP = AveragePrecision(task='binary').cuda()
        de_st_detect_AUROC = AUROC(task='binary').cuda()
        seg_IAPS = IAPS().cuda()
        seg_AUPRO = AUPRO().cuda()
        seg_AUROC = AUROC(task='binary').cuda()
        seg_AP = AveragePrecision(task='binary').cuda()
        seg_detect_AUROC = AUROC(task='binary').cuda()

        for sample_batched in tqdm(dataloader, desc="Evaluating"):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()
            filenames = sample_batched["file"]
            
            # 模型推理
            output_segmentation, output_de_st, output_de_st_list, contrast = model(img)
            
            # 调整输出大小
            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            output_de_st = F.interpolate(
                output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
            )

            # 对每个图像保存原始缺陷图的彩色热力图
            for i, filename in enumerate(filenames):
                # 解析文件名 (格式: defect_type + index)
                if len(filename) > 3:
                    defect_type = filename[:-3]
                    file_idx = filename[-3:]
                else:
                    defect_type = "unknown"
                    file_idx = filename
                
                # 创建输出目录
                output_colormap_dir = os.path.join("./output", "output_colormap", category, defect_type)
                os.makedirs(output_colormap_dir, exist_ok=True)
                
                # 获取原始异常图
                anomaly_map_np = output_segmentation[i].squeeze().detach().cpu().numpy()
                
                # 创建彩色热力图
                colormap = create_colormap(anomaly_map_np)
                
                # 保存彩色热力图
                cv2.imwrite(os.path.join(output_colormap_dir, f"{file_idx}.png"), colormap)
            
            # 特殊类别处理：应用SAM掩码
            if category in SPECIAL_CATEGORIES:
                for i, filename in enumerate(filenames):
                    # 跳过good类别
                    if "good" in filename:
                        continue
                        
                    # 解析文件名 (格式: defect_type + index)
                    if len(filename) > 3:
                        defect_type = filename[:-3]
                        file_idx = filename[-3:]
                    else:
                        defect_type = "unknown"
                        file_idx = filename
                    
                    # 加载SAM掩码
                    sam_mask_path = os.path.join(
                        args.mvtec_path,
                        category,
                        "sam_results",
                        defect_type,
                        f"{file_idx}.png"
                    )
                    
                    if os.path.exists(sam_mask_path):
                        # 加载并预处理SAM掩码
                        sam_mask = Image.open(sam_mask_path).convert("L")
                        sam_mask = sam_mask.resize(RESIZE_SHAPE, Image.NEAREST)
                        sam_mask = np.array(sam_mask) / 255.0
                        sam_mask = torch.from_numpy(sam_mask).unsqueeze(0).unsqueeze(0).float().cuda()
                        
                        # 应用SAM掩码到分割输出
                        output_segmentation[i:i+1] = output_segmentation[i:i+1] * (sam_mask <= 0.45).float()
                        
                        # 保存处理后的缺陷掩码（彩色热力图）
                        output_front_dir = os.path.join("./output", "output_front_colormap", category, defect_type)
                        os.makedirs(output_front_dir, exist_ok=True)
                        
                        # 获取应用掩码后的异常图
                        anomaly_map_np = output_segmentation[i].squeeze().detach().cpu().numpy()
                        
                        # 创建彩色热力图
                        colormap = create_colormap(anomaly_map_np)
                        
                        # 保存彩色热力图
                        cv2.imwrite(os.path.join(output_front_dir, f"{file_idx}.png"), colormap)
                        
                        # 创建并保存融合图像
                        concat_dir = os.path.join(
                            "/root/.vscode-server/data/eval-master-PseudoMaskContrastLoss-true_mask/output",
                            "concat_front_results",
                            category,
                            defect_type
                        )
                        os.makedirs(concat_dir, exist_ok=True)
                        
                        # 反归一化原始图像
                        mean = np.array(NORMALIZE_MEAN).reshape(1, 1, 3)
                        std = np.array(NORMALIZE_STD).reshape(1, 1, 3)
                        original_img_denorm = img[i].permute(1, 2, 0).cpu().numpy()
                        original_img_denorm = np.clip((original_img_denorm * std + mean) * 255, 0, 255).astype(np.uint8)
                        
                        # 创建融合图像
                        fusion_img = original_img_denorm.copy()
                        for c in range(3):
                            fusion_img[:, :, c] = np.where(
                                (sam_mask.squeeze().cpu().numpy() > 0.5) & (anomaly_map_np > 0.3),
                                colormap[:, :, c] * 0.7 + original_img_denorm[:, :, c] * 0.3,
                                original_img_denorm[:, :, c]
                            )
                        
                        # 保存融合图像
                        cv2.imwrite(os.path.join(concat_dir, f"{file_idx}.png"), fusion_img)
                    else:
                        print(f"Warning: SAM mask not found at {sam_mask_path}")

            # 指标计算 (使用处理后的output_segmentation)
            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            
            # 分割指标
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )
            
            # 去噪指标
            output_de_st_sample, _ = torch.sort(
                output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
            )
            output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)

            # 更新指标
            de_st_IAPS.update(output_de_st, mask)
            de_st_AUPRO.update(output_de_st, mask)
            de_st_AP.update(output_de_st.flatten(), mask.flatten())
            de_st_AUROC.update(output_de_st.flatten(), mask.flatten())
            de_st_detect_AUROC.update(output_de_st_sample, mask_sample)

            seg_IAPS.update(output_segmentation, mask)
            seg_AUPRO.update(output_segmentation, mask)
            seg_AP.update(output_segmentation.flatten(), mask.flatten())
            seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            seg_detect_AUROC.update(output_segmentation_sample, mask_sample)

        # 计算并记录指标
        iap_de_st, iap90_de_st = de_st_IAPS.compute()
        aupro_de_st, ap_de_st, auc_de_st, auc_detect_de_st = (
            de_st_AUPRO.compute(),
            de_st_AP.compute(),
            de_st_AUROC.compute(),
            de_st_detect_AUROC.compute(),
        )
        iap_seg, iap90_seg = seg_IAPS.compute()
        aupro_seg, ap_seg, auc_seg, auc_detect_seg = (
            seg_AUPRO.compute(),
            seg_AP.compute(),
            seg_AUROC.compute(),
            seg_detect_AUROC.compute(),
        )

        # 记录指标
        visualizer.add_scalar("DeST_IAP", iap_de_st, global_step)
        visualizer.add_scalar("DeST_IAP90", iap90_de_st, global_step)
        visualizer.add_scalar("DeST_AUPRO", aupro_de_st, global_step)
        visualizer.add_scalar("DeST_AP", ap_de_st, global_step)
        visualizer.add_scalar("DeST_AUC", auc_de_st, global_step)
        visualizer.add_scalar("DeST_detect_AUC", auc_detect_de_st, global_step)

        visualizer.add_scalar("DeSTSeg_IAP", iap_seg, global_step)
        visualizer.add_scalar("DeSTSeg_IAP90", iap90_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AUPRO", aupro_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AP", ap_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AUC", auc_seg, global_step)
        visualizer.add_scalar("DeSTSeg_detect_AUC", auc_detect_seg, global_step)

        print("Eval at step", global_step)
        print("================================")
        print("Denoising Student-Teacher (DeST)")
        print("pixel_AUC:", round(float(auc_de_st), 4))
        print("pixel_AP:", round(float(ap_de_st), 4))
        print("PRO:", round(float(aupro_de_st), 4))
        print("image_AUC:", round(float(auc_detect_de_st), 4))
        print("IAP:", round(float(iap_de_st), 4))
        print("IAP90:", round(float(iap90_de_st), 4))
        print()
        print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
        print("pixel_AUC:", round(float(auc_seg), 4))
        print("pixel_AP:", round(float(ap_seg), 4))
        print("PRO:", round(float(aupro_seg), 4))
        print("image_AUC:", round(float(auc_detect_seg), 4))
        print("IAP:", round(float(iap_seg),4))
        print("IAP90:", round(float(iap90_seg), 4))
        print()

        # 保存评估结果到CSV
        save_dir = os.path.join('./output', f"{args.datasource}", category)
        os.makedirs(output_colormap_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'evaluation_results.csv')

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Eval at step", global_step])
            writer.writerow(["Denoising Student-Teacher (DeST)"])
            writer.writerow(["pixel_AUC", round(float(auc_de_st), 4)])
            writer.writerow(["pixel_AP", round(float(ap_de_st), 4)])
            writer.writerow(["PRO", round(float(aupro_de_st), 4)])
            writer.writerow(["image_AUC", round(float(auc_detect_de_st), 4)])
            writer.writerow(["IAP", round(float(iap_de_st), 4)])
            writer.writerow(["IAP90", round(float(iap90_de_st), 4)])
            writer.writerow([])
            writer.writerow(["Segmentation Guided Denoising Student-Teacher (DeSTSeg)"])
            writer.writerow(["pixel_AUC", round(float(auc_seg), 4)])
            writer.writerow(["pixel_AP", round(float(ap_seg), 4)])
            writer.writerow(["PRO", round(float(aupro_seg), 4)])
            writer.writerow(["image_AUC", round(float(auc_detect_seg), 4)])
            writer.writerow(["IAP", round(float(iap_seg), 4)])
            writer.writerow(["IAP90", round(float(iap90_seg), 4)])

        # 重置指标
        de_st_IAPS.reset()
        de_st_AUPRO.reset()
        de_st_AUROC.reset()
        de_st_AP.reset()
        de_st_detect_AUROC.reset()
        seg_IAPS.reset()
        seg_AUPRO.reset()
        seg_AUROC.reset()
        seg_AP.reset()
        seg_detect_AUROC.reset()

        return ap_seg, auc_seg

def test(args, category):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"DeSTSeg_MVTec_test_{category}"
    log_dir = os.path.join(args.log_path, run_name)
    
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    visualizer = SummaryWriter(log_dir=log_dir)

    model = EnhancedDeSTSeg(category=category).cuda()
    
    # 加载共享编码器参数
    shared_ckpt = os.path.join(args.checkpoint_path, "shared_components", "shared_encoder.pth")
    if os.path.exists(shared_ckpt):
        model.shared_encoder.load_state_dict(torch.load(shared_ckpt))
    
    # 加载类别特定参数
    class_ckpt = os.path.join(args.checkpoint_path, f"{category}_params", "class_specific.pth")
    if os.path.exists(class_ckpt):
        class_params = torch.load(class_ckpt)
        model.decoder.load_state_dict(class_params['decoder'])
        model.seg_net.load_state_dict(class_params['seg_net'])

    return evaluate(args, category, model, visualizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--datasource", type=str, default="MVTEC")

    parser.add_argument("--mvtec_path", type=str, default="/root/.vscode-server/data/Datasets/mvtec/")
    parser.add_argument("--test_path", type=str, default="/root/.vscode-server/data/Datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="/root/.vscode-server/data/Datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_models/")

    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_6000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=SPECIAL_CATEGORIES)
    args = parser.parse_args()

    # 确保类别名称使用下划线而不是空格
    obj_list = args.category
    for i, obj in enumerate(obj_list):
        if " " in obj:
            obj_list[i] = obj.replace(" ", "_")
    
    # 检查类别是否在SPECIAL_CATEGORIES中
    obj_list = [obj for obj in obj_list if obj in SPECIAL_CATEGORIES]
    
    if not obj_list:
        print("No valid categories to process. Exiting.")
        exit()

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(f"Testing {obj}")
            test(args, obj)