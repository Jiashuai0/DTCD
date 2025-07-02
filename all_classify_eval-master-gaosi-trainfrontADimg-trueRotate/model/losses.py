import torch
import torch.nn.functional as F


def cosine_similarity_loss(output_de_st_list):
    loss = 0
    for instance in output_de_st_list:
        _, _, h, w = instance.shape
        loss += torch.sum(instance) / (h * w)
    return loss

def contrast_loss(output_de_st_list):
    loss = 0
    current_batchsize = output_de_st_list[0][0].shape[0]

    target = -torch.ones(current_batchsize).to('cuda')
    contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)
    loss = contrast(output_de_st_list[0][0].view(output_de_st_list[0][0].shape[0], -1), output_de_st_list[1][0].view(output_de_st_list[1][0].shape[0], -1), target = target) + \
        contrast(output_de_st_list[0][1].view(output_de_st_list[0][1].shape[0], -1), output_de_st_list[1][1].view(output_de_st_list[1][1].shape[0], -1), target = target)+ \
        contrast(output_de_st_list[0][2].view(output_de_st_list[0][2].shape[0], -1), output_de_st_list[1][2].view(output_de_st_list[1][2].shape[0], -1), target = target)

    return loss


#threshold 是高斯函数的中心点（默认 0.5），当余弦相似度等于它时损失最大。<控制损失的中心在哪>
#sigma 控制高斯函数的宽度（标准差），越小表示函数越“尖”<控制损失的宽度>
def gaussian_contrastive_loss(output_de_st_list, threshold=0.0, sigma=0.2):#其中的可调节参数就是threshold和sigma
    """
    Gaussian contrastive loss:
    y = exp(-((cos_sim - threshold)^2) / (2 * sigma^2))
    
    Parameters:
        output_de_st_list: List of two lists, each with 3 feature maps (student, teacher)
        threshold: Center of Gaussian curve, where loss is maximal
        sigma: Controls the width of the Gaussian curve
    
    Returns:
        total_loss: Scalar loss value
    """
    loss = 0  #初始化总损失为 0
    eps = 1e-8  # 常用于数值稳定（可删除或备用）
    for i in range(3):  # 对三层特征图（如 shallow/middle/deep）分别计算损失
        student_feat = output_de_st_list[0][i]#取出学生网络在第 i 层的特征图
        teacher_feat = output_de_st_list[1][i]#取出教师网络在第 i 层的特征图

        # 将每张特征图 (B, C, H, W) reshape 为 (B, C*H*W)，便于计算每一张图的向量相似性
        student_flat = student_feat.view(student_feat.shape[0], -1)
        teacher_flat = teacher_feat.view(teacher_feat.shape[0], -1)

        # 对每个样本（按 dim=1）进行 L2 归一化，使其范数为 1
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)

        # 点积计算每一对样本之间的余弦相似度（因为向量已归一化，点积就是 cos 值）;得到一个形如 (B,) 的向量，表示 batch 中每个样本的相似度
        cos_sim = torch.sum(student_norm * teacher_norm, dim=1)

        # Gaussian loss: y = exp(-((x - t)^2) / (2 * sigma^2))	应用高斯函数计算损失值
        gaussian_loss = torch.exp(-((cos_sim - threshold) ** 2) / (2 * sigma ** 2))
        loss += gaussian_loss.mean()

    #return loss / 3  # 将三层损失取平均，作为最终的 loss 输出
    return loss   # 将三层损失之和，作为最终的 loss 输出



#使用真实掩码来当作对比损失的掩码
def masked_contrast_loss(output_de_st_list, mask):#mask表示传入的原始图像所对应的真实掩码，来对mask进行下采样得到对比损失所对应的掩码
    """
    Args:
        output_de_st_list: list 包含三个元素:
            - outputs_teacher_aug: 合成图经过教师网络的输出 (3层特征图)
            - outputs_student_aug: 合成图经过学生网络的输出 (3层特征图)
            - outputs_teacher: 正常图像经过教师网络的输出 (3层特征图)
        alpha: 阈值，用于生成掩码

    Returns:
        loss: 总对比损失
    """
    outputs_teacher_aug = output_de_st_list[0]
    outputs_student_aug = output_de_st_list[1]
    outputs_teacher = output_de_st_list[2]

    loss = 0
    contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)

    for i in range(3):  # 遍历三层特征
        feat_teacher_aug = outputs_teacher_aug[i]     # (B, C, H, W)
        feat_student_aug = outputs_student_aug[i]     # (B, C, H, W)
        feat_teacher = outputs_teacher[i]             # (B, C, H, W)

        # 下采样mask到当前特征图尺寸
        B, C, H_feat, W_feat = feat_teacher_aug.shape
        mask_i = F.interpolate(
            mask,
            size=(H_feat, W_feat),
            mode="bilinear",
            align_corners=False,
        )
        # 关键修正：正常区域作为正样本 (mask=0 → target=1)，异常区域作为负样本 (mask=1 → target=-1)
        mask_i = torch.where(mask_i >= 0.5, 1.0, 0.0)  # 原mask：1=异常，0=正常
        target = (1.0 - mask_i) * 2 - 1  # 转换为：正常=1，异常=-1
        target = target.view(-1)  # 展平为(B*H_feat*W_feat)


        # 生成空间掩码：小于 alpha 的设为 1，其余设为 -1
        #mask = torch.where(feature_diff < alpha, torch.ones_like(feature_diff), -torch.ones_like(feature_diff))  # (B, 1, H, W)

        # 展平特征和掩码
        #B, C, H, W = feat_teacher_aug.shape
        feat_teacher_flat = feat_teacher_aug.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        feat_student_flat = feat_student_aug.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        #mask_flat = mask.view(B, -1)  # (B, H*W)
        
        # 合并批次和空间维度
        feat_teacher_flat = feat_teacher_flat.reshape(-1, C)  # (B*H*W, C)
        feat_student_flat = feat_student_flat.reshape(-1, C)  # (B*H*W, C)
        #mask_flat = mask_flat.reshape(-1)  # (B*H*W,1) #注意，每个样本对应一个标签，如果想要每个像素对应一个标签，就要把每个像素当成一个样本，所以就把B*H*W当成样本数，每个样本对应一个标签

        # 计算对比损失
        # 这里的 target 也可以是 (B, C*H*W)，即每个元素的 target
        loss += contrast(feat_teacher_flat, feat_student_flat, target = target)

    return loss  # 返回总对比损失


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)
