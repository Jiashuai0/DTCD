import argparse
import os
import shutil
import warnings
import glob
import random
import math
import numpy as np
from PIL import Image
import cv2
#from data.data_utils import perlin_noise
from data.data_utils import perlin_noise
import imgaug.augmenters as iaa

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torch.utils.data import Dataset, DataLoader  # 修改这行导入
from torchvision import transforms  # 添加这行导入
import torchvision.transforms.functional as TF  # 添加这行导入
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from eval_new_new import evaluate
from model.destseg_old import ClassSpecificDecoder, ClassSpecificSegNet, EnhancedDeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss, gaussian_contrastive_loss

warnings.filterwarnings("ignore")

# 定义需要特殊处理的类别（只保留前景区域异常）
SPECIAL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'hazelnut', 
    'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor'
]

# Perlin噪声生成函数 (从文档1中提取)
def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: cv2.resize(
        np.repeat(
            np.repeat(
                gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],
                d[0], 
                axis=0
            ),
            d[1],
            axis=1,
        ),
        dsize=(shape[1], shape[0]),
    )
    
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )

rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
augmenters = [
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.Posterize(),
    iaa.Invert(),
    iaa.pillike.Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45))
]

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([
        augmenters[aug_ind[0]],
        augmenters[aug_ind[1]],
        augmenters[aug_ind[2]]
    ])
    return aug

def perlin_noise_with_sam_mask(image, dtd_image, sam_mask, aug_prob=1.0):
    """生成带SAM前景掩码的Perlin噪声图像"""
    image = np.array(image, dtype=np.float32)
    dtd_image = np.array(dtd_image, dtype=np.uint8)
    aug = randAugmenter()
    dtd_image = aug(image=dtd_image)
    shape = image.shape[:2]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = random.randint(min_perlin_scale, max_perlin_scale)
    t_y = random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y

    # 生成Perlin噪声
    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))
    perlin_noise = rot(images=perlin_noise)
    perlin_noise = np.expand_dims(perlin_noise, axis=2)
    threshold = 0.5
    
    # 应用SAM前景掩码
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )
    
    # 将SAM掩码调整为与噪声相同形状并二值化
    sam_mask = np.array(sam_mask, dtype=np.float32)
    sam_mask = cv2.resize(sam_mask, (shape[1], shape[0]))
    sam_mask = np.where(sam_mask > 127, 0.0, 1.0)  # 前景为1，背景为0
    
    # 应用SAM掩码到噪声
    masked_perlin_thr = perlin_thr * sam_mask[:, :, np.newaxis]

    img_thr = dtd_image.astype(np.float32) * masked_perlin_thr / 255.0
    image = image / 255.0

    beta = random.random() * 0.8
    image_aug = (
        image * (1 - masked_perlin_thr) + 
        (1 - beta) * img_thr + 
        beta * image * (masked_perlin_thr)
    )
    image_aug = image_aug.astype(np.float32)

    no_anomaly = random.random()
    if no_anomaly > aug_prob:
        return image, np.zeros_like(masked_perlin_thr)
    else:
        msk = (masked_perlin_thr).astype(np.float32)
        msk = msk.transpose(2, 0, 1)
        return image_aug, msk

# 自定义数据集类，支持SAM掩码
class MVTecDatasetWithSAM(Dataset):
    def __init__(
        self,
        is_train,
        classname,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.classname = classname
        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate

        # 设置数据集路径
        self.mvtec_dir = mvtec_dir
        self.dtd_dir = dtd_dir
        self.sam_dir = os.path.join(os.path.dirname(os.path.dirname(mvtec_dir)), "sam_results_train", "good")
        
        if is_train:
            self.mvtec_paths = sorted(glob.glob(os.path.join(mvtec_dir, "*.png")))
            self.dtd_paths = sorted(glob.glob(os.path.join(dtd_dir, "*", "*.jpg")))
            self.sam_paths = sorted(glob.glob(os.path.join(self.sam_dir, "*.png")))
        else:
            self.mvtec_paths = sorted(glob.glob(os.path.join(mvtec_dir, "*", "*.png")))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)
    
            fill_color = (114, 114, 114)
            fill_color_mask = 0  # 掩码的填充颜色（黑色）
            # 保存原始图像和掩码用于可视化
            img_origin_np = np.array(image)
            sam_mask_np = None
            # 加载SAM掩码
            sam_mask = Image.new("L", self.resize_shape, color=0)  # 默认创建全黑掩码
            if self.classname in SPECIAL_CATEGORIES and index < len(self.sam_paths):
                sam_mask = Image.open(self.sam_paths[index]).convert("L")
                sam_mask = sam_mask.resize(self.resize_shape, Image.BILINEAR)
                sam_mask_np = np.array(sam_mask)
            
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
                # 对SAM掩码应用相同旋转
                sam_mask = sam_mask.rotate(
                    degree, fillcolor=fill_color_mask, resample=Image.NEAREST
                )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
                # 对SAM掩码应用相同旋转
                sam_mask = sam_mask.rotate(
                    degree, fillcolor=fill_color_mask, resample=Image.NEAREST
                )

            # 保存旋转后的SAM掩码用于后续处理和可视化
            sam_mask_np = np.array(sam_mask)  # 形状为 (H, W)，值在0-255之间

            # 特殊类别使用SAM掩码生成异常
            if self.classname in SPECIAL_CATEGORIES:
                
                # 使用带SAM掩码的Perlin噪声
                aug_image, aug_mask = perlin_noise_with_sam_mask(
                    np.array(image), 
                    np.array(dtd_image),
                    np.array(sam_mask),
                    aug_prob=1.0
                )
                
                # 转换为PIL图像
                aug_image = (aug_image * 255).astype(np.uint8)
                aug_image = Image.fromarray(aug_image)
            else:
                # 普通Perlin噪声
                aug_image, aug_mask = perlin_noise(
                    np.array(image), 
                    np.array(dtd_image),
                    aug_prob=1.0
                )
                aug_image = Image.fromarray((aug_image * 255).astype(np.uint8))
            # 处理掩码 (1, H, W) -> (H, W)
            aug_mask = (aug_mask[0] * 255).astype(np.uint8)
            aug_mask = Image.fromarray(aug_mask)

            aug_image = self.final_preprocessing(aug_image)
            image = self.final_preprocessing(image)
            
            # 返回用于可视化的数据
            return {
                "img_aug": aug_image,
                "img_origin": image,
                "mask": torch.from_numpy(np.array(aug_mask) / 255.0).float(),
                # 新增：返回用于可视化的原始数据
                "img_origin_np": np.array(image),    # 原始图像数组
                "aug_img_np": np.array(aug_image),   # 合成异常图像数组
                "sam_mask_np": sam_mask_np,        # SAM掩码数组
                "aug_mask_np": np.array(aug_mask)    # 合成异常掩码数组
            }
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            file = base_dir+file_name.split(".")[0]
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"img": image, "mask": mask, "file": file}

def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = EnhancedDeSTSeg(category).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.seg_net.seg_net.res.parameters(), "lr": args.lr_res},
            {"params": model.seg_net.seg_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.decoder.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # 使用自定义数据集
    dataset = MVTecDatasetWithSAM(
        is_train=True,
        classname=category,
        mvtec_dir=os.path.join(args.mvtec_path, category, "train", "good"),
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0
    flag = True
    best_pauc, best_pap = 0, 0
    
    # 创建保存图像的目录
    os.makedirs("./output/augimg", exist_ok=True)

    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            # 修改后
            mask = sample_batched["mask"].cuda().unsqueeze(1)  # 添加通道维度

            mask_origin = sample_batched["mask"].cuda()
            mask_origin = torch.where(mask_origin < 0.5, torch.zeros_like(mask_origin), torch.ones_like(mask_origin))

            if global_step < args.de_st_steps:
                model.shared_encoder.eval()
                model.decoder.train()
                model.seg_net.eval()
            else:
                model.shared_encoder.eval()
                model.decoder.eval()
                model.seg_net.train()

            output_segmentation, output_de_st, output_de_st_list, contrast = model(
                img_aug, img_origin
            )

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            gaussian_contrastive_loss_val = gaussian_contrastive_loss(contrast)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = (cosine_loss_val + 0.3 * gaussian_contrastive_loss_val) / 1.3
                total_loss_val.backward()
                de_st_optimizer.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            # 修改后的保存图像代码
            if global_step % 100 == 0:
                save_dir = f"./output/augimg/step_{global_step}"
                os.makedirs(save_dir, exist_ok=True)
                print(f"Saving sample images to {save_dir} at step {global_step}")
    
                batch_size = args.bs
                for i in range(batch_size):
                    # 获取当前样本的类别
                    category = sample_batched["category"][i] if "category" in sample_batched else "unknown"
        
                    # 创建类别子目录
                    category_dir = os.path.join(save_dir, category)
                    os.makedirs(category_dir, exist_ok=True)
        
                    # 获取各种图像数据
                    img_origin = sample_batched["img_origin_np"][i].cpu().numpy().astype(np.uint8)  # 原始图像
                    aug_img = sample_batched["aug_img_np"][i].cpu().numpy().astype(np.uint8)  # 合成异常图像
                    sam_mask = sample_batched["sam_mask_np"][i].cpu().numpy().astype(np.uint8)  # SAM掩码
                    aug_mask = sample_batched["aug_mask_np"][i].cpu().numpy().astype(np.uint8)  # 合成异常掩码
        
                    # 保存原始图像
                    img_path = os.path.join(category_dir, f"{i}_origin.png")
                    cv2.imwrite(img_path, cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR))
                    
                    # 保存合成异常图像
                    aug_img_path = os.path.join(category_dir, f"{i}_aug.png")
                    cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    # 保存SAM掩码
                    sam_mask_path = os.path.join(category_dir, f"{i}_sam_mask.png")
                    cv2.imwrite(sam_mask_path, sam_mask)
                    
                    # 保存合成异常掩码
                    aug_mask_path = os.path.join(category_dir, f"{i}_aug_mask.png")
                    cv2.imwrite(aug_mask_path, aug_mask)
                
                print(f"Saved {batch_size} sample images to {save_dir}")


            if global_step % args.eval_per_steps == 0:
                tmp_pauc, tmp_pap = evaluate(args, category, model, visualizer, global_step)
                if tmp_pauc + tmp_pap > best_pauc + best_pap:
                    best_pauc, best_pap = tmp_pauc, tmp_pap
                    # 保存共享编码器
                    os.makedirs("saved_models/shared_components", exist_ok=True)
                    torch.save(model.shared_encoder.state_dict(), 
                            "saved_models/shared_components/shared_encoder.pth")
                    # 保存类别特定组件
                    class_dir = f"saved_models/{category}_params"
                    os.makedirs(class_dir, exist_ok=True)
                    torch.save({
                        "decoder": model.decoder.state_dict(),
                        "seg_net": model.seg_net.state_dict()
                    }, f"{class_dir}/class_specific.pth")
                    
            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                    )

            if global_step >= args.steps:
                flag = False
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="/root/.vscode-server/data/Datasets/mvtec/")
    parser.add_argument("--test_path", type=str, default="/root/.vscode-server/data/Datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="/root/.vscode-server/data/Datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_shared_encoder", type=float, default=0.1)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.11)
    parser.add_argument("--lr_seghead", type=float, default=0.012)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--de_st_steps", type=int, default=1000)
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--datasource", type=str, default="old")
    parser.add_argument("--custom_training_category", action="store_true", default=False)
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--slight_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        for category in (no_rotation_category + slight_rotation_category + rotation_category):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]

    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=0)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)