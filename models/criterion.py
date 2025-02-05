# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import einops
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils import is_distributed, distributed_world_size
from typing import Tuple
import torch
from torch import nn, Tensor


class VideoReIDCircleLoss(nn.Module):
    def __init__(self, m: float = 0.25, gamma: float = 256, _lambda: float = 0.01):
        super(VideoReIDCircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self._lambda = _lambda  # 缩放因子
        self.soft_plus = nn.Softplus()

    def forward(self, feature_dict):
        all_features = []
        all_labels = []
        for idx, features in feature_dict.items():
            all_features.append(features)
            all_labels.extend([idx] * features.size(0))

        features = torch.cat(all_features, dim=0)
        labels = torch.tensor(all_labels, device=features.device)

        # L2 归一化特征
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.matmul(features, features.t())

        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        sim_matrix = sim_matrix - torch.eye(sim_matrix.size(0), device=sim_matrix.device) * 2

        pos_mask = label_matrix.float()
        neg_mask = (1 - pos_mask) * (1 - torch.eye(sim_matrix.size(0), device=sim_matrix.device))

        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask

        # 使用 clamp 来限制值的范围，提高数值稳定性
        alpha_p = torch.clamp_min(-pos_sim.detach() + 1 + self.m, min=0)
        alpha_n = torch.clamp_min(neg_sim.detach() + self.m, min=0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = (-self.gamma) * alpha_p * (pos_sim - delta_p)
        logit_n = self.gamma * alpha_n * (neg_sim - delta_n)

        # 使用 log_sum_exp 技巧来提高数值稳定性
        def log_sum_exp(x, dim):
            max_val = torch.max(x, dim=dim, keepdim=True)[0]
            return max_val + torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))

        loss = self.soft_plus(log_sum_exp(logit_n, dim=1) + log_sum_exp(logit_p, dim=1))

        num_valid_samples = torch.sum(torch.sum(pos_mask, dim=1) > 0)

        # 应用缩放因子并确保损失非负
        loss = self._lambda * torch.clamp(loss.sum() / num_valid_samples, min=0)

        return loss


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)

    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix], normed_feature.shape[1]


class CircleLoss(nn.Module):
    def __init__(self, m: float = 0.8, gamma: float = 80, _lambda: float = 1) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self._lambda = _lambda

    def forward(self, sp: Tensor, sn: Tensor, size_normed_feature) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        loss = self._lambda * torch.clamp(loss / size_normed_feature, min=0)

        return loss

    def extra_repr(self):
        return f'm={self.m}, gamma={self.gamma}, lambda={self._lambda}'


# 使用示例

class IDCriterion(nn.Module):
    def __init__(self, weight: float, gpu_average: bool):
        super().__init__()
        self.weight = weight
        self.gpu_average = gpu_average
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.IDLoss = CircleLoss()

    def forward(self, outputs, targets, emb_reid):
        assert len(outputs) == 1, f"ID Criterion is only supported bs=1, but get bs={len(outputs)}"
        outputs = einops.rearrange(outputs, "b n c -> (b n) c")
        # targets = einops.rearrange(targets, "b n c -> (b n) c")
        targets = targets.squeeze(0)
        # ce_loss = self.ce_loss(outputs, targets).sum()
        # Average:
        num_ids = len(outputs)
        num_ids = torch.as_tensor([num_ids], dtype=torch.float, device=outputs.device)
        if self.gpu_average:
            if is_distributed():
                torch.distributed.all_reduce(num_ids)
            num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        reid_loss = self.IDLoss(*convert_label_to_similarity(outputs, targets))

        # num_boxes = len(emb_reid)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs.device)
        # if self.gpu_average:
        #     if is_distributed():
        #         torch.distributed.all_reduce(num_boxes)
        #     num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        # reid_loss = self.IDLoss(emb_reid)
        return reid_loss / num_ids
        # return ce_loss / num_ids
        # return (ce_loss * 0.8 / num_ids + reid_loss * 0.2 / num_ids)


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # 初始化 log sigma 参数
        # self.log_sigma_det = nn.Parameter(-4.85 * torch.ones(1))
        self.log_sigma_det = nn.Parameter(-4.66 * torch.ones(1))
        self.log_sigma_reid = nn.Parameter(-2 * torch.ones(1))

    def forward(self, loss_det, loss_reid):
        # if loss_reid > 10:
        #     loss = (torch.exp(-self.log_sigma_det) * loss_det +
        #             torch.exp(-self.log_sigma_reid) * loss_reid +
        #             self.log_sigma_det + self.log_sigma_reid)
        # else:
        #     loss = loss_det + torch.exp(-self.log_sigma_reid*self.log_sigma_det) * loss_reid

        loss = loss_det + loss_reid
        return loss



def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        gpu_average=config["ID_LOSS_GPU_AVERAGE"]
    )


def build_multi_task_loss():
    return MultiTaskLoss()


# 使用示例
if __name__ == "__main__":
    criterion = VideoReIDCircleLoss()

    # 模拟输入数据
    feature_dict = {
        0: torch.randn(5, 61),  # ID 0 有5帧
        1: torch.randn(3, 61),  # ID 1 有3帧
        2: torch.randn(4, 61),  # ID 2 有4帧
    }

    loss = criterion(feature_dict)
    print(f"Loss: {loss.item()}")
