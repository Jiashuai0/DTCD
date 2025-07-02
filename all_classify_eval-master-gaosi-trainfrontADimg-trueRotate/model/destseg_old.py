import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18
from model.model_utils_old import ASPP, BasicBlock, l2_normalize, make_layer

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models

class FineTunedClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        #self.base_model = timm.create_model("resnet18", pretrained=False, features_only=False)
        self.base_model = models.resnet18(weights=False)
        self.base_model.fc = nn.Linear(512, num_classes)
        self._load_pretrained("/root/.vscode-server/data/all_classify_eval-master/logs/fine_tuning/fold_9_best.pth")#微调后的分类网络的参数保存的位置

    def _load_pretrained(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
         # 为所有键添加 'base_model.' 前缀
        new_state_dict = {f'base_model.{k}': v for k, v in state_dict.items()}
        # 加载调整后的参数，允许部分不匹配（如有必要）
        self.load_state_dict(new_state_dict, strict=True)
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)




class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)

"""
class StudentNet(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

        _,self.bn = resnet18(pretrained=True)
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        # x = x4
        if not self.ed:
            return (x1, x2, x3)
        x = [x1,x2,x3]
        x = self.bn(x)
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)"""

"""
#各个类别共享的encoder模块
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.encoder = timm.create_model(
            "resnet18", pretrained=False, features_only=True, out_indices=[1, 2, 3, 4]
        )
        

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        #x = [x1,x2,x3]
        #x = self.bn(x)#OCBE模块
        return [x1, x2, x3]"""

#各个类别共享的encoder模块
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用与TeacherNet相同的预训练设置
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3]  # 输出三个层级特征
        )
        # 冻结所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        return [x1, x2, x3]  # 保持与原来一致的输出结构



#各个类别独有的decoder模块
class ClassSpecificDecoder(nn.Module):
    def __init__(self, ed=True):
        super().__init__()

        _,self.bn = resnet18(pretrained=True)#师兄加的，直接引用OCBE模块

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

    def forward(self, x_list):
        # 与原StudentNet相同的decoder逻辑
        x1, x2, x3= x_list
        x = self.bn(x_list)#OCBE模块
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)



class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

#各个类别的语义分割模块(SegmentationNet)
class ClassSpecificSegNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.seg_net = SegmentationNet(inplanes)
    
    def forward(self, x):
        return self.seg_net(x)

"""
class DeSTSeg(nn.Module):
    def __init__(self, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img_aug,img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]

        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug)
        ]
        # output_to_visualize = outputs_student_aug[0]
        # image_to_visualize = output_to_visualize[0, 0].detach().cpu().numpy()  # 形状 (64, 64)

        # # 插值到 256x256
        # image_to_visualize = F.interpolate(
        #     torch.tensor(image_to_visualize).unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度
        #     size=(256, 256),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze().numpy()  # 移除多余的维度

        # # 归一化到 0-255 并转换为 uint8
        # image_to_visualize = (image_to_visualize - np.min(image_to_visualize)) / (np.max(image_to_visualize) - np.min(image_to_visualize)) * 255
        # image_to_visualize = image_to_visualize.astype(np.uint8)

        # # 保存可视化
        # output_path = 'output_visualization.jpg'
        # cv2.imwrite(output_path, image_to_visualize)
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )
        #         #         # 创建保存目录
        # save_dir = os.path.join('saved_new',category)
        # os.makedirs(save_dir, exist_ok=True)
        # to_pil = transforms.ToPILImage()
        # # 对通道维度进行平均
        # rgb_images = []  # 创建一个空列表以存储每个批次的 RGB 图像
        # feature_map = torch.mean(output, dim=1)
        # for i in range(feature_map.shape[0]):  # 遍历每个批次
        #     feature = feature_map[i].unsqueeze(0) # 形状变为 [64, 64]
        #     # feature_upsampled = torch.nn.functional.interpolate(feature.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        #     # 归一化到 [0, 1]
        #     feature_upsampled = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)  
            
        #     # 转换为 PIL 图像
        #     rgb_image = to_pil(feature_upsampled)  
        #     rgb_image.save(os.path.join(save_dir, f'de_st_{fill_name[i]}.png'))  # 保存为 PNG 文件
        #     rgb_images.append(rgb_image)
        # print(f"Saved {feature_map.shape[0]} images to '{save_dir}' directory.")

        # output_test = torch.cat(
        #     [
        #         F.interpolate(
        #             output_s,
        #             size=outputs_student_aug[0].size()[2:],  # 上采样到相同的空间大小
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #         for output_s in outputs_student_aug  # 只遍历学生模型的输出
        #     ],
        #     dim=1,  # 在通道维度上连接
        # )

        # # rgb_images = []  # 创建一个空列表以存储每个批次的 RGB 图像
        # # 对通道维度进行平均
        # feature_map = torch.mean(output_test, dim=1)
        # for i in range(feature_map.shape[0]):  # 遍历每个批次
        #     feature = feature_map[i].unsqueeze(0) # 形状变为 [64, 64]
        #     # feature_upsampled = torch.nn.functional.interpolate(feature.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        #     # 归一化到 [0, 1]
        #     feature_upsampled = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)  
            
        #     # 转换为 PIL 图像
        #     rgb_image = to_pil(feature_upsampled)  
        #     # rgb_images.append(rgb_image)
        #     rgb_image.save(os.path.join(save_dir, f'recon_{fill_name[i]}.png'))  # 保存为 PNG 文件

        # print(f"Saved {feature_map.shape[0]} images to '{save_dir}' directory.")
        # output_to_visualize = outputs_student_aug[0]
        # image_to_visualize = output_to_visualize[0, 0].detach().cpu().numpy()  # 形状 (64, 64)

        # # 插值到 256x256
        # image_to_visualize = F.interpolate(
        #     torch.tensor(image_to_visualize).unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度
        #     size=(256, 256),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze().numpy()  # 移除多余的维度

        # # 归一化到 0-255 并转换为 uint8
        # image_to_visualize = (image_to_visualize - np.min(image_to_visualize)) / (np.max(image_to_visualize) - np.min(image_to_visualize)) * 255
        # image_to_visualize = image_to_visualize.astype(np.uint8)

        # # 保存可视化
        # output_path = 'output_visualization.jpg'
        # cv2.imwrite(output_path, image_to_visualize)

        output_segmentation = self.segmentation_net(output)

        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]
        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]


        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)
        contrast = [outputs_teacher_aug,outputs_student_aug]
        # return output_segmentation, output_de_st, output_de_st_list,contrast,rgb_images
        return output_segmentation, output_de_st, output_de_st_list,contrast"""


class EnhancedDeSTSeg(nn.Module):
    def __init__(self, category):
        super().__init__()
        # 教师网络（保持原状）
        self.teacher_net = TeacherNet()
        
        # 共享组件
        self.shared_encoder = SharedEncoder()
        #self._load_shared_encoder()  # 新增共享编码器加载

        # 类别特定组件
        self.decoder = ClassSpecificDecoder()
        self.seg_net = ClassSpecificSegNet()
        self._load_class_specific(category)  # 修改为分类加载
        """# 加载类别特定参数
        self._load_category_params(category)"""

    """def _load_category_params(self, category):
        # 加载指定类别的decoder和seg_net参数
        ckpt_path = f"saved_models/{category}_params.pth"
        if os.path.exists(ckpt_path):
            params = torch.load(ckpt_path)
            self.decoder.load_state_dict(params['decoder'])
            self.seg_net.load_state_dict(params['seg_net'])"""
    """def _load_shared_encoder(self):
        #加载全局共享编码器
        shared_path = "saved_models/shared_components/shared_encoder.pth"
        if os.path.exists(shared_path):
            self.shared_encoder.load_state_dict(torch.load(shared_path))"""


    def _load_shared_encoder(self):
        """加载共享编码器的预训练权重"""
        shared_ckpt_path = "saved_models/shared_components/shared_encoder.pth"
        if os.path.exists(shared_ckpt_path):
            self.shared_encoder.load_state_dict(torch.load(shared_ckpt_path))
        else:
            print(f"Warning: Shared encoder checkpoint not found at {shared_ckpt_path}")
            

    def load_shared_encoder(self, ckpt_path):
        self.shared_encoder.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded shared encoder from {ckpt_path}")


    def _load_class_specific(self, category):
        """加载类别特定参数"""
        class_path = f"saved_models/{category}_params/class_specific.pth"
        if os.path.exists(class_path):
            params = torch.load(class_path)
            self.decoder.load_state_dict(params['decoder'])
            self.seg_net.load_state_dict(params['seg_net'])

    def save_category_params(self, category):
        os.makedirs("saved_models", exist_ok=True)
        torch.save({
            'decoder': self.decoder.state_dict(),
            'seg_net': self.seg_net.state_dict(),
        }, f"saved_models/{category}_params.pth")

    def forward(self, img_aug, img_origin=None):
        self.teacher_net.eval()
        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        #教师网络1：输入合成异常图像，得到的各个尺度特征：T3,T2,T1
        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]

        

        # 共享特征提取
        shared_features = self.shared_encoder(img_aug)#教师网络输入合成异常特征图，得到的各个尺度特征：T3,T2,T1
        
        # 类别特定处理
        decoder_out = self.decoder(shared_features)

        outputs_student_aug = [#学生网络输入合成异常特征图，得到的各个尺度特征：SD3,SD2,SD1
            l2_normalize(output_s) for output_s in decoder_out
        ]

        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )



        output_segmentation = self.seg_net(output)
        outputs_student = outputs_student_aug
        #教师网络2：输入正常图像，得到的各个尺度特征：T3,T2,T1
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]
        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]


        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)
        #contrast = [outputs_teacher_aug,outputs_student_aug]
        contrast = [outputs_teacher_aug,outputs_student_aug,outputs_teacher]#我们在contrast列表中加入了将正常图和对应的合成异常图
        # return output_segmentation, output_de_st, output_de_st_list,contrast,rgb_images
        return output_segmentation, output_de_st, output_de_st_list,contrast
