import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
#old
from data.mvtec_dataset import MVTecDataset

# #new
# from data.mvtec_dataset_np import MVTecDataset
from eval_new_new import evaluate
# # # 改动
# from model.destseg_new import DeSTSeg

# 原来
from model.destseg_old import ClassSpecificDecoder, ClassSpecificSegNet, EnhancedDeSTSeg

from model.losses import cosine_similarity_loss, focal_loss, l1_loss,contrast_loss,masked_contrast_loss,gaussian_contrastive_loss
warnings.filterwarnings("ignore")

"""
def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    ##old
    dataset = MVTecDataset(
        is_train=True,

        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    ##new
    # dataset = MVTecDataset(
    #     is_train=True,
    #     classname=category,
    #     mvtec_dir=args.mvtec_path + category + "/train/good/",
    #     resize_shape=RESIZE_SHAPE,
    #     normalize_mean=NORMALIZE_MEAN,
    #     normalize_std=NORMALIZE_STD,
    #     dtd_dir=args.dtd_path,
    # )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0

    flag = True
    best_pauc,best_pap = 0,0
    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation, output_de_st, output_de_st_list,contrast = model(
                img_aug, img_origin
            )

            # output_segmentation, output_de_st, output_de_st_list = model(
            #     img_aug, img_origin
            # )
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
            contrast_loss_val = contrast_loss(contrast)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = (cosine_loss_val+0.3*contrast_loss_val)/1.3

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

            if global_step % args.eval_per_steps == 0:
                tmp_pauc,tmp_pap = evaluate(args, category, model, visualizer, global_step)
                if tmp_pauc+tmp_pap > best_pauc+best_pap:
                    best_pauc,best_pap = tmp_pauc,tmp_pap
                    torch.save(
                        model.state_dict(), os.path.join(args.checkpoint_path, run_name + "_best.pckl")
                    )
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
                break"""




"""
def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))
    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    # 初始化共享组件
    shared_encoder = SharedEncoder().cuda()
    shared_ckpt = os.path.join(args.checkpoint_path, "shared_encoder.pth")
    
    # 加载共享编码器预训练权重（如果存在）
    if os.path.exists(shared_ckpt):
        shared_encoder.load_state_dict(torch.load(shared_ckpt))
        print(f"Loaded shared encoder from {shared_ckpt}")

    # 初始化类别特定组件
    decoder = ClassSpecificDecoder().cuda()
    seg_net = ClassSpecificSegNet().cuda()
    
    # 创建完整模型
    model = DeSTSeg(
        shared_encoder=shared_encoder,
        class_decoder=decoder,
        seg_net=seg_net
    ).cuda()


    # 优化器设置（包含共享编码器+类别特定组件）
    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.seg_net.res.parameters(), "lr": args.lr_res},#没有这个参数？把lr_seg改为这个？？？？？？？？？？
            {"params": model.seg_net.head.parameters(), "lr": args.lr_seghead}, #没有这个参数？把lr_seg改为这个？？？？？？？？？？
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    dest_optimizer = torch.optim.SGD(
        [
        {"params": model.shared_encoder.parameters(), "lr": args.lr_shared},
        {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        #{"params": seg_net.parameters(), "lr": args.lr_seg}
        ], 
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # 保持原有的数据加载逻辑
    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
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

    # 初始化最佳指标
    best_pauc, best_pap = 0, 0
    global_step = 0
    flag = True

    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            dest_optimizer.zero_grad()
            # 数据准备
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            # 阶段切换逻辑保持原始设定
            if global_step < args.de_st_steps:
                #shared_encoder.eval()
                shared_encoder.train()
                decoder.train()
                #seg_net.train()
                seg_net.eval()
            else:
                #shared_encoder.train()
                #decoder.train()
                shared_encoder.eval()
                decoder.eval()
                seg_net.train()

            # 改进的前向传播流程
            with torch.set_grad_enabled(global_step >= args.de_st_steps):
                # 共享编码器提取特征
                features = shared_encoder(img_aug)
                
                # 类别特定解码
                decoder_out = decoder(features)
                
                # 分割网络输出
                output_segmentation = seg_net(decoder_out)

                # 保持原有的去噪学生网络逻辑
                output_de_st, output_de_st_list, contrast = dest_network(
                    img_origin, img_aug, shared_encoder
                )

            # 保持原有的损失计算逻辑
            if global_step < args.de_st_steps:
                loss = cosine_loss(output_de_st_list) + 0.3*contrast_loss(contrast)
            else:
                focal_loss_val = focal_loss(output_segmentation, mask)
                l1_loss_val = l1_loss(output_segmentation, mask)
                loss = focal_loss_val + l1_loss_val

            # 反向传播
            loss.backward()
            seg_optimizer.step()

            # 保持原有的日志记录逻辑
            visualizer.add_scalar("total_loss", loss, global_step)
            if global_step % args.log_per_steps == 0:
                print(f"Step {global_step} Loss: {loss.item():.4f}")

            # 评估与保存逻辑
            if global_step % args.eval_per_steps == 0:
                # 构建临时推理模型
                tmp_model = {
                    'shared_encoder': shared_encoder,
                    'decoder': decoder,
                    'seg_net': seg_net,
                    'dest_network': dest_network
                }
                
                # 执行评估
                tmp_pauc, tmp_pap = evaluate(args, category, tmp_model, visualizer, global_step)
                
                # 保存最佳模型
                if (tmp_pauc + tmp_pap) > (best_pauc + best_pap):
                    best_pauc, best_pap = tmp_pauc, tmp_pap
                    torch.save({
                        'decoder': decoder.state_dict(),
                        'seg_net': seg_net.state_dict()
                    }, f"{args.checkpoint_path}/{category}_best.pth")
                    
                    # 保存共享编码器
                    torch.save(shared_encoder.state_dict(), shared_ckpt)

            global_step += 1
            if global_step >= args.steps:
                flag = False
                break

# 评估函数需要相应调整
def evaluate(args, category, model_dict, visualizer, global_step):
    model_dict['shared_encoder'].eval()
    model_dict['decoder'].eval()
    model_dict['seg_net'].eval()
    
    with torch.no_grad():
        # 保持原有评估逻辑，使用组合模型进行推理
        output_seg = model_dict['seg_net'](
            model_dict['decoder'](
                model_dict['shared_encoder'](img)
            )
        )
        
        # ... 后续评估指标计算保持不变 ..."""





def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    # 修改模型初始化
    model = EnhancedDeSTSeg(category).cuda()  # 使用增强版模型

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
            #{"params": model.shared_encoder.parameters(), "lr": args.lr_shared_encoder},#共享encoder部分的学习率
            #{"params": model.ClassSpecificDecoder.parameters(), "lr": args.lr_de_st},#各个类别所特有的decoder部分的学习率
            {"params": model.decoder.parameters(), "lr": args.lr_de_st},#各个类别所特有的decoder部分的学习率
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    ##old
    dataset = MVTecDataset(
        is_train=True,

        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    ##new
    # dataset = MVTecDataset(
    #     is_train=True,
    #     classname=category,
    #     mvtec_dir=args.mvtec_path + category + "/train/good/",
    #     resize_shape=RESIZE_SHAPE,
    #     normalize_mean=NORMALIZE_MEAN,
    #     normalize_std=NORMALIZE_STD,
    #     dtd_dir=args.dtd_path,
    # )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0

    flag = True
    best_pauc,best_pap = 0,0
    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()

            mask_origin = sample_batched["mask"].cuda()#保存和原始图像大小一样的真实掩码mask，传入masked_contrast_loss损失函数中，作为对比损失函数的掩码
            mask_origin = torch.where(mask_origin < 0.5, torch.zeros_like(mask_origin), torch.ones_like(mask_origin))#将真实掩码


            if global_step < args.de_st_steps:
                #model.student_net.train()
                model.shared_encoder.eval()
                model.decoder.train()
                #model.segmentation_net.eval()
                model.seg_net.eval()
            else:
                #model.student_net.eval()
                #model.segmentation_net.train()
                model.shared_encoder.eval()
                model.decoder.eval()
                model.seg_net.train()

            output_segmentation, output_de_st, output_de_st_list,contrast = model(
                img_aug, img_origin
            )

            # output_segmentation, output_de_st, output_de_st_list = model(
            #     img_aug, img_origin
            # )
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
            #contrast_loss_val = contrast_loss(contrast)
            #contrast_loss = masked_contrast_loss(contrast,mask_origin)#
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            gaussian_contrastive_loss_val = gaussian_contrastive_loss(contrast)##师兄添加的
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = (cosine_loss_val+0.3*gaussian_contrastive_loss_val)/1.3

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

            if global_step % args.eval_per_steps == 0:
                tmp_pauc,tmp_pap = evaluate(args, category, model, visualizer, global_step)
                if tmp_pauc+tmp_pap > best_pauc+best_pap:
                    best_pauc,best_pap = tmp_pauc,tmp_pap
                    # 保存共享编码器（全局）
                    """shared_ckpt_dir = os.path.join(args.checkpoint_path, "shared_components")
                    os.makedirs(shared_ckpt_dir, exist_ok=True)
                    torch.save(model.shared_encoder.state_dict(), 
                            os.path.join(shared_ckpt_dir, "shared_encoder.pth"))"""
                    os.makedirs("saved_models/shared_components", exist_ok=True)  # 新增目录创建
                    torch.save(model.shared_encoder.state_dict(), 
                            "saved_models/shared_components/shared_encoder.pth")
                    # 保存类别特定组件（每个类别单独目录）
                    """class_ckpt_dir = os.path.join(args.checkpoint_path, f"{category}_params")
                    os.makedirs(class_ckpt_dir, exist_ok=True)
                    torch.save({
                        "decoder": model.decoder.state_dict(),
                        "seg_net": model.seg_net.state_dict()
                    }, os.path.join(class_ckpt_dir, "class_specific.pth"))"""
                    class_dir = f"saved_models/{category}_params"
                    os.makedirs(class_dir, exist_ok=True)  # 新增目录创建
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
    # parser.add_argument("--lr_de_st", type=float, default=0.2)
    # parser.add_argument("--lr_res", type=float, default=0.04)
    # parser.add_argument("--lr_seghead", type=float, default=0.005)
    parser.add_argument("--lr_shared_encoder", type=float, default=0.1)#共享encoder部分的学习率
    parser.add_argument("--lr_de_st", type=float, default=0.4)#各个类别所特有的decoder部分的学习率
    parser.add_argument("--lr_res", type=float, default=0.11)#分割网络的残差块部分
    parser.add_argument("--lr_seghead", type=float, default=0.012)#分割网络的ASPP部分
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference
    parser.add_argument("--datasource",type=str,default="old")
    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
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
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)