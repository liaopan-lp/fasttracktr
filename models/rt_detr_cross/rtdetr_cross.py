"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import copy
from utils import box_ops
from utils.nested_tensor import NestedTensor, tensor_list_to_nested_tensor
from utils.utils import inverse_sigmoid, accuracy, interpolate, is_distributed, distributed_world_size
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .backbone import build_backbone
from .backbone_rt import build_backbone_presnet
from .matcher_for_group import build_matcher
from .hybrid_encoder import build_hybird_encoder
from .rtdetr_decoder_cross import build_rt_transformer
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from models.criterion import build_multi_task_loss

from .rtdetr_criterion import SetCriterion
import os
import math

__all__ = ['RTDETRCROSS', ]


class RTDETRCROSS(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, multi_task_loss: nn.Module = None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        # self.trajectory_encoder = trajectory_encoder
        self.multi_scale = multi_scale
        self.multi_task_loss = multi_task_loss

    def forward(self, x, targets=None, max_gt_num=0, trajectory_inputs=None):
        x = x.tensors
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            sz = int(sz)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)

        # 放decoder里面吧
        # if trajectory_inputs is not None:
        #     src = trajectory_inputs['src']
        #     mask = trajectory_inputs['mask']
        #     trajectory_input = self.trajectory_encoder(src, mask, feat=x)
        #     trajectory_inputs = [trajectory_input, mask]

        x = self.decoder(x, targets, max_gt_num=max_gt_num, trajectory_inputs=trajectory_inputs)

        return x

    def pos2posemb(self, pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=pos.dtype, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        posemb = pos[..., None] / dim_t
        posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
        return posemb

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


# class SetCriterion(nn.Module):
#     """ This class computes the loss for DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#
#     def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, group_num=1):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha
#         self.gamma = 2
#         self.group_num = group_num
#
#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         src_logits = outputs['pred_logits']
#
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#
#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
#         target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
#
#         target_classes_onehot = target_classes_onehot[:, :, :-1]
#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
#                   src_logits.shape[1]
#         losses = {'loss_ce': loss_ce}
#
#         if log:
#             # TODO this should probably be a separate loss, not hacked in this one here
#             losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
#         return losses
#
#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices, num_boxes):
#         """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
#         This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
#         """
#         pred_logits = outputs['pred_logits']
#         device = pred_logits.device
#         tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
#         # Count the number of predictions that are NOT "no-object" (which is the last class)
#         card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         losses = {'cardinality_error': card_err}
#         return losses
#
#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#
#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#
#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes
#
#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses
#
#     def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
#         ious = torch.diag(ious).detach()
#
#         src_logits = outputs['pred_logits']
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#         target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
#
#         target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
#         target_score_o[idx] = ious.to(target_score_o.dtype)
#         target_score = target_score_o.unsqueeze(-1) * target
#
#         pred_score = F.sigmoid(src_logits).detach()
#         weight = self.focal_alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
#
#         loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
#         loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
#         return {'loss_vfl': loss}
#
#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs
#
#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#
#         src_masks = outputs["pred_masks"]
#
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = tensor_list_to_nested_tensor([t["masks"] for t in targets]).decompose()
#         target_masks = target_masks.to(src_masks)
#
#         src_masks = src_masks[src_idx]
#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)
#
#         target_masks = target_masks[tgt_idx].flatten(1)
#
#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses
#
#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx
#
#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx
#
#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'cardinality': self.loss_cardinality,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks,
#             'vfl': self.loss_labels_vfl,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
#
#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
#         group_num = self.group_num if self.training else 1
#
#         indices_for_loss = []
#
#         # Retrieve the matching between the outputs of the last layer and the targets
#         init_indices = self.matcher(outputs_without_aux, targets, group_num=group_num)
#         indices_for_loss.append(init_indices)
#
#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets) * group_num
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_distributed():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()
#
#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, init_indices, num_boxes, **kwargs))
#
#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets, group_num=group_num)
#                 # indices_for_loss.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs['log'] = False
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#         if 'enc_outputs' in outputs:
#             enc_outputs = outputs['enc_outputs']
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 bt['labels'] = torch.zeros_like(bt['labels'])
#             if os.environ.get('IPDB_SHILONG_DEBUG') == 'INFO':
#                 import ipdb
#                 ipdb.set_trace()
#             indices = self.matcher(enc_outputs, bin_targets, group_num=group_num)
#             # indices_for_loss.append(indices)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs['log'] = False
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_enc': v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#
#         return losses, init_indices


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def build_rt_detr_cross(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone_presnet(args)

    # transformer = RTDETR(args)
    encoder = build_hybird_encoder(args,backbone.out_channels)
    decoder = build_rt_transformer(args)
    multi_task_loss = build_multi_task_loss()
    model = RTDETRCROSS(
        backbone,
        encoder,
        decoder,
        multi_scale=args.multi_scale,
        multi_task_loss=multi_task_loss
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, group_num=args.group_num)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, None
