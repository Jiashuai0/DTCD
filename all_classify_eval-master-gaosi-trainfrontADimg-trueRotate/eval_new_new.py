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

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.destseg_old import  EnhancedDeSTSeg, FineTunedClassifier
from model.metrics import AUPRO, IAPS
import matplotlib.pyplot as plt
import numpy as np
import csv
warnings.filterwarnings("ignore")

class ClassifierWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = FineTunedClassifier()
        
    def forward(self, x):
        return self.classifier(x)

def evaluate(args, category, model, visualizer, global_step=0):
    model.eval()

    # 加载分类器并设置为评估模式
    classifier = ClassifierWrapper().cuda()
    #classifier_path = os.path.join(args.checkpoint_path, "fold_9_best.pth")  # 假设分类器权重路径
    
    #刚去掉的呜呜呜
    #classifier.load_state_dict(torch.load("/root/.vscode-server/data/all_classify_eval-master/logs/fine_tuning/fold_9_best.pth"))
    
    #新加的
    state_dict = torch.load("/root/.vscode-server/data/all_classify_eval-master/logs/fine_tuning/fold_9_best.pth")

    # 调整键前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f'classifier.base_model.{key}'
        new_state_dict[new_key] = value

    classifier.load_state_dict(new_state_dict, strict=True)


    classifier.eval()

    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=os.path.join(args.mvtec_path, category, "test/"),
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
        print(f"Test dataset size for {category}: {len(dataset)}")  # 调试输出
        if len(dataset) == 0:
            raise ValueError(f"No test samples found for {category}!")
    
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        if len(dataloader) == 0:
            raise ValueError(f"Dataloader for {category} is empty!")

        # 初始化指标计算器
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
            if sample_batched["img"] is None:
                print("Warning: Empty batch encountered.")
                continue
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()
            filenames = sample_batched["file"]
            
            """
            # 预测当前batch的类别
            cls_logits = classifier(img)
            cls_pred = torch.argmax(cls_logits, dim=1)
            predicted_category = ALL_CATEGORY[cls_pred[0].item()]  # 假设整个batch属于同一类别
            model._load_category_params(predicted_category)  # 动态加载模型参数"""

            # 模型前向传播
            output_segmentation, output_de_st, output_de_st_list, contrast = model(img)

            # 调整输出大小
            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            output_de_st = F.interpolate(
                output_de_st, size=mask.shape[2:], mode="bilinear", align_corners=False
            )

            # 计算样本级指标
            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_segmentation_sample = torch.topk(
                output_segmentation.view(output_segmentation.size(0), -1),
                k=args.T,
                dim=1
            )[0].mean(dim=1)
            output_de_st_sample = torch.topk(
                output_de_st.view(output_de_st.size(0), -1),
                k=args.T,
                dim=1
            )[0].mean(dim=1)

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

        # 计算结果并记录
        iap_de_st, iap90_de_st = de_st_IAPS.compute()
        aupro_de_st = de_st_AUPRO.compute()
        ap_de_st = de_st_AP.compute()
        auc_de_st = de_st_AUROC.compute()
        auc_detect_de_st = de_st_detect_AUROC.compute()

        iap_seg, iap90_seg = seg_IAPS.compute()
        aupro_seg = seg_AUPRO.compute()
        ap_seg = seg_AP.compute()
        auc_seg = seg_AUROC.compute()
        auc_detect_seg = seg_detect_AUROC.compute()

        # 记录到TensorBoard（示例保留部分关键指标）
        visualizer.add_scalar(f"{category}/DeST_IAP", iap_de_st, global_step)
        visualizer.add_scalar(f"{category}/DeST_AUPRO", aupro_de_st, global_step)
        visualizer.add_scalar(f"{category}/Seg_IAP", iap_seg, global_step)
        visualizer.add_scalar(f"{category}/Seg_AUPRO", aupro_seg, global_step)

        # 保存结果到CSV
        save_dir = os.path.join('./output', args.datasource, category)
        os.makedirs(save_dir, exist_ok=True)
        # 添加类型转换确保为Python float
        iap_de_st = iap_de_st.item() if hasattr(iap_de_st, 'item') else iap_de_st
        iap_seg = iap_seg.item() if hasattr(iap_seg, 'item') else iap_seg

        # 构建结果字典
        results = {
            "DeST": {
                "pixel_AUC": auc_de_st.item(),
                "pixel_AP": ap_de_st.item(),
                "PRO": aupro_de_st.item(),
                "image_AUC": auc_detect_de_st.item(),
                "IAP": iap_de_st,
                "IAP90": iap90_de_st
            },
            "DeSTSeg": {
                "pixel_AUC": auc_seg.item(),
                "pixel_AP": ap_seg.item(),
                "PRO": aupro_seg.item(),
                "image_AUC": auc_detect_seg.item(),
                "IAP": iap_seg,
                "IAP90": iap90_seg
            }
        }




        with open(os.path.join(save_dir, 'results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "DeST", "Seg"])
            # 修改行：直接使用浮点数值（IAP指标）
            writer.writerow(["IAP", iap_de_st, iap_seg])  
            # 保留其他指标.item()（假设这些指标返回Tensor）
            writer.writerow(["AUPRO", aupro_de_st.item(), aupro_seg.item()]) 
            writer.writerow(["AP", ap_de_st.item(), ap_seg.item()])
            writer.writerow(["AUROC", auc_de_st.item(), auc_seg.item()])
            writer.writerow(["Detect_AUROC", auc_detect_de_st.item(), auc_detect_seg.item()])

        return auc_seg.item(), ap_seg.item()

def test(args, category):
    # 初始化模型
    model = EnhancedDeSTSeg(category=category).cuda()
    # 加载共享编码器参数（路径需与训练保存路径一致）
    #shared_ckpt = os.path.join(args.checkpoint_path, "shared_components/shared_encoder.pth")
    #model.load_shared_encoder(shared_ckpt)  # 使用新添加的方法
    
     # 加载类别特定参数
    class_ckpt = os.path.join(args.checkpoint_path, f"{category}_params/class_specific.pth")
    if os.path.exists(class_ckpt):
        params = torch.load(class_ckpt)
        model.decoder.load_state_dict(params['decoder'])
        model.seg_net.load_state_dict(params['seg_net'])


     # 加载该类别特定参数（如果存在）
    model._load_class_specific(category)

    # 初始化可视化工具
    log_dir = os.path.join(args.log_path, f"test_{category}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 执行评估
    evaluate(args, category, model, writer)
    
    # 关闭写入器
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mvtec_path", type=str, default="/root/.vscode-server/data/Datasets/mvtec/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_models/")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--category", nargs="+", default=ALL_CATEGORY)
    parser.add_argument("--datasource", type=str, default="old")  # 新增此行
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    for obj in args.category:
        print(f"Testing {obj}")
        test(args, obj)