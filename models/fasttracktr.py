# Copyright (c) PanLiao. All Rights Reserved.
# About:
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from torch.utils.checkpoint import checkpoint
from .seq_decoder import SeqDecoder
from .deformable_detr.deformable_detr import build as build_deformable_detr
from .dab_deformable_detr.dab_deformable_detr import build_dab_deformable_detr
from .rt_detr_cross.rtdetr_cross import build_rt_detr_cross
from structures.instances import Instances
from structures.args import Args
from utils.utils import batch_iterator, combine_detr_outputs
from utils.utils import labels_to_one_hot
from collections import deque
from structures.ordered_set import OrderedSet
import math
import random
import einops


class FastTrackTr(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.num_id_vocabulary = config["DETR_NUM_QUERIES"]  # how many id words
        self.training_num_id = config["DETR_NUM_QUERIES"] if "DETR_NUM_QUERIES" not in config else config[
            "DETR_NUM_QUERIES"]
        self.num_classes = config["NUM_CLASSES"]
        self.max_temporal_length = config["MAX_TEMPORAL_LENGTH"] if "MAX_TEMPORAL_LENGTH" in config \
            else config["MAX_TEMPORAL_PE_LENGTH"]

        # DETR Framework:
        self.detr_framework = config["DETR_FRAMEWORK"]
        # Backbone:
        self.backbone_type = config["BACKBONE"]
        self.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
        self.backbone_dilation = config["DILATION"]
        # DETR settings:
        self.detr_num_queries = config["DETR_NUM_QUERIES"]
        self.detr_num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
        self.detr_aux_loss = config["DETR_AUX_LOSS"]
        self.detr_with_box_refine = config["DETR_WITH_BOX_REFINE"]
        self.detr_two_stage = config["DETR_TWO_STAGE"]
        self.detr_masks = config["DETR_MASKS"]
        self.detr_hidden_dim = config["DETR_HIDDEN_DIM"]
        self.detr_position_embedding = config["DETR_PE"]
        self.detr_nheads = config["DETR_NUM_HEADS"]
        self.detr_enc_layers = config["DETR_ENC_LAYERS"]
        self.detr_dec_layers = config["DETR_DEC_LAYERS"]
        self.detr_dropout = config["DETR_DROPOUT"]
        self.detr_dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
        self.detr_dec_n_points = config["DETR_DEC_N_POINTS"]
        self.detr_enc_n_points = config["DETR_ENC_N_POINTS"]
        self.detr_cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
        self.detr_bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
        self.detr_giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
        self.detr_set_cost_class = config["DETR_CLS_LOSS_COEF"] if "DETR_SET_COST_CLASS" not in config else config[
            "DETR_SET_COST_CLASS"]
        self.detr_set_cost_bbox = config["DETR_BBOX_LOSS_COEF"] if "DETR_SET_COST_BBOX" not in config else config[
            "DETR_SET_COST_BBOX"]
        self.detr_set_cost_giou = config["DETR_GIOU_LOSS_COEF"] if "DETR_SET_COST_GIOU" not in config else config[
            "DETR_SET_COST_GIOU"]
        self.detr_focal_alpha = config["DETR_FOCAL_ALPHA"]
        self.detr_group_num = config["GROUP_NUM"]

        self.device = config["DEVICE"]

        self.only_detr = config["TRAIN_STAGE"] == "only_detr"

        # Prepare args for detr
        detr_args = Args()
        detr_args.num_classes = self.num_classes
        detr_args.device = self.device
        detr_args.num_queries = self.detr_num_queries
        detr_args.num_feature_levels = self.detr_num_feature_levels
        detr_args.aux_loss = self.detr_aux_loss
        detr_args.with_box_refine = self.detr_with_box_refine
        detr_args.two_stage = self.detr_two_stage
        detr_args.hidden_dim = self.detr_hidden_dim
        detr_args.backbone = self.backbone_type
        detr_args.lr_backbone = self.lr_backbone
        detr_args.dilation = self.backbone_dilation
        detr_args.masks = self.detr_masks
        detr_args.position_embedding = self.detr_position_embedding
        detr_args.nheads = self.detr_nheads
        detr_args.enc_layers = self.detr_enc_layers
        detr_args.dec_layers = self.detr_dec_layers
        detr_args.dim_feedforward = self.detr_dim_feedforward
        detr_args.dropout = self.detr_dropout
        detr_args.dec_n_points = self.detr_dec_n_points
        detr_args.enc_n_points = self.detr_enc_n_points
        detr_args.cls_loss_coef = self.detr_cls_loss_coef
        detr_args.bbox_loss_coef = self.detr_bbox_loss_coef
        detr_args.giou_loss_coef = self.detr_giou_loss_coef
        detr_args.focal_alpha = self.detr_focal_alpha
        # Three hack implementation:
        detr_args.set_cost_class = self.detr_set_cost_class
        detr_args.set_cost_bbox = self.detr_set_cost_bbox
        detr_args.set_cost_giou = self.detr_set_cost_giou
        # for rt-detr backbone
        detr_args.backbone_depth = config["depth"]
        detr_args.variant = config["variant"]
        detr_args.freeze_at = config["freeze_at"]
        detr_args.return_idx = config["return_idx"]
        detr_args.freeze_norm = config["freeze_norm"]
        detr_args.pretrained = config["pretrained"]
        detr_args.num_stages = config["num_stages"]

        # encoder 里的
        detr_args.multi_scale = config["multi_scale"]

        detr_args.group_num = self.detr_group_num
        self.multi_task_loss = None

        if self.detr_framework == "Deformable-DETR":
            # DETR model and criterion:
            self.detr, self.detr_criterion, _ = build_deformable_detr(detr_args)
        elif self.detr_framework == "DAB-Deformable-DETR":
            detr_args.num_patterns = 0
            detr_args.random_refpoints_xy = False
            self.detr, self.detr_criterion, _ = build_dab_deformable_detr(detr_args)
            # TODO: We will upload the DAB-DETR code soon.
        elif self.detr_framework == "RT-DETR":
            self.detr, self.detr_criterion, _, self.multi_task_loss = build_rt_detr_cross(detr_args)
        else:
            raise RuntimeError(f"Unknown DETR framework: {self.detr_framework}.")
        # ID Label Criterion:
        self.id_criterion = nn.CrossEntropyLoss()
        self.id_loss_weight = config["ID_LOSS_WEIGHT"]

    @torch.no_grad()
    def update_track(
            self,
            pred_id_words,
            trajectory_history: deque[Instances],
            num_id_vocabulary: int,
            ids_to_results: dict,
            current_id: int,
            id_deque: OrderedSet,
            id_thresh: float = 0.1,
            newborn_thresh: float = 0.5,
            inference_ensemble: int = 0,
    ):
        pass

    @torch.no_grad()
    def update_id(
            self,
            pred_id_words,
            trajectory_history: deque[Instances],
            num_id_vocabulary: int,
            ids_to_results: dict,
            current_id: int,
            id_deque: OrderedSet,
            id_thresh: float = 0.1,
            newborn_thresh: float = 0.5,
            inference_ensemble: int = 0,
    ):
        """
        :param trajectory_history: Historical trajectories.
        :param num_id_vocabulary: Number of ID vocabulary, K in the paper.
        :param ids_to_results: Mapping from ID word index to ID label in tracker files.
        :param current_id: Current next ID label of tracker files.
        :param id_deque: OrderedSet of ID words, may be recycled.
        :param id_thresh: ID threshold.
        :param newborn_thresh: Newborn threshold,
                               only the conf higher than this threshold will be considered as a newborn target.
        :param inference_ensemble: Ensemble times for inference.
        :return:
        """
        deque_max_length = trajectory_history.maxlen
        trajectory_history_list = list(trajectory_history)
        trajectory = trajectory_history_list[:-1]
        current = trajectory_history_list[-1:]

        # NEED TO KNOW:
        # 1. "ids" is the final ID words for current frame, it is a list.
        #    If a target does not have a corresponding ID word, it will be assigned as -1 in "ids".
        # 2. "new_id" is the ID words that need to be assigned to the new targets, also a list.
        # 3. "current" is the objects in the current frame.

        n_targets_in_frames = [len(_) for _ in trajectory_history_list]
        num_history_tokens, num_current_tokens = sum(n_targets_in_frames[:-1]), sum(n_targets_in_frames[-1:])
        if num_history_tokens == 0:  # no history tokens
            ids = [-1] * num_current_tokens
        elif num_current_tokens == 0:  # no current tokens
            ids = []
            return ids, trajectory_history, ids_to_results, current_id, id_deque, None  # directly return
        else:  # normal process:
            trajectory_id_set = set(torch.cat([_.ids for _ in trajectory_history_list[:-1]], dim=0).cpu().tolist())
            # # Seq Decoding:
            # pred_id_words, _ = self.seq_decoder(
            #     track_seqs=[trajectory_history_list],
            #     inference_ensemble=inference_ensemble,
            # )
            pass

            if isinstance(pred_id_words, torch.Tensor):
                id_confs = torch.softmax(pred_id_words, dim=-1)  # [1, N, K + 1]
                # id_confs = id_confs  # [N, K + 1]
            else:
                assert isinstance(pred_id_words, list)
                # id_confs = [torch.softmax(_, dim=1) for _ in pred_id_words]
                id_confs = [_ for _ in pred_id_words]
                id_confs = [_[0] for _ in id_confs]
                _ensemble_n = len(id_confs)
                id_confs = torch.stack(id_confs, dim=0)  # [T, N, K + 1]
                id_confs = torch.sum(id_confs, dim=0)  # [N, K + 1]
                id_confs = id_confs / _ensemble_n
                id_confs = torch.softmax(id_confs, dim=-1)  # [N, K + 1]
                pass

            ids = list()
            newborn_repeat = id_confs[:, -1:].repeat(1, len(id_confs) - 1)
            extended_id_confs = torch.cat((id_confs, newborn_repeat), dim=-1)
            match_rows, match_cols = linear_sum_assignment(1 - extended_id_confs.cpu())  # simple and efficient
            for _ in range(len(match_rows)):
                _id = match_cols[_]
                if _id not in trajectory_id_set:
                    ids.append(-1)
                elif _id >= num_id_vocabulary:
                    ids.append(-1)
                elif id_confs[match_rows[_], _id].item() < id_thresh:
                    ids.append(-1)
                else:
                    ids.append(_id)

            # Here is a customized implementation for ID assignment,
            # as an alternative to the Hungarian algorithm.
            # However, the Hungarian algorithm is more efficient and simpler (off-the-shelf package).
            # These two implementations only brings a slight difference in performance.
            # In our practice, < 0.3 HOTA on DanceTrack, < 0.1 HOTA on MOT17.
            # each_id_max_confs = torch.max(id_confs, dim=0).values
            # ids = list()
            # for i in range(len(id_confs)):
            #     target_id_confs, target_ids = torch.topk(id_confs[i], k=len(id_confs[0]))
            #     target_id = None    # final target ID word index
            #     for c in range(len(target_id_confs)):
            #         _id, _conf = target_ids[c].item(), target_id_confs[c].item()
            #         if _id == num_id_vocabulary:        # newborn
            #             target_id = -1
            #             break
            #         if _conf < id_thresh:
            #             break                           # early stop
            #         if _conf < each_id_max_confs[_id].item():
            #             continue                        # not the best choice
            #         else:
            #             if _id == num_id_vocabulary:
            #                 target_id = -1
            #             elif _id not in trajectory_id_set:
            #                 target_id = -1
            #             else:
            #                 target_id = _id
            #                 each_id_max_confs[_id] = 1.01   # hack implementation, avoid double assign
            #             break
            #     if target_id is None:
            #         target_id = -1
            #     ids.append(target_id)

        # Update the ID deque:
        for _id in ids:
            if _id != -1:
                id_deque.add(_id)

        # Filter the newborn targets, True means marked as newborn but not reach the newborn threshold:
        newborn_neg_filter = ((torch.tensor(ids).to(current[0].confs.device) == -1)
                              & (current[0].confs <= newborn_thresh).reshape(-1, ))

        if torch.sum(~newborn_neg_filter) > num_id_vocabulary:
            # The legal objects are too many, we need to filter out some of them.
            # Warning: This should not happen in normal cases.
            #          If it happens, you may increase the ID vocabulary size.
            print(f"[Warning!] There are too many objects, N={torch.sum(~newborn_neg_filter)}. ")
            already_ids_num = torch.sum(torch.tensor(ids) != -1)
            newborn_index = torch.tensor(ids).to(current[0].confs.device) == -1
            confs = current[0].confs.reshape(-1, ) * newborn_index.to(float)
            newborn_num_in_legal = num_id_vocabulary - already_ids_num
            index = torch.topk(confs, k=newborn_num_in_legal, dim=0).indices
            newborn_neg_filter_from_topk = torch.tensor(ids).to(current[0].confs.device) == -1
            newborn_neg_filter_from_topk[index] = False
            legal_newborn_neg_filter = newborn_neg_filter | newborn_neg_filter_from_topk
            newborn_neg_filter = legal_newborn_neg_filter
            print(f"[Warning!] Because the newborn objects are too many, "
                  f"we only keep {newborn_num_in_legal} newborn objects with highest confs. "
                  f"Already assigned {already_ids_num} IDs. "
                  f"Now we have {torch.sum(~newborn_neg_filter)} IDs.")

        # Just a check!
        assert torch.sum(~newborn_neg_filter) <= num_id_vocabulary, f"Too many IDs: {torch.sum(~newborn_neg_filter)}."

        # Remove the illegal newborn targets (conf < newborn_thresh):
        ids = torch.tensor(ids)[~newborn_neg_filter.cpu()].tolist()
        current[0] = current[0][~newborn_neg_filter]

        num_new_id = ids.count(-1)  # how many new ID words need to be assigned

        if num_new_id > 0:  # assign new ID words
            id_deque_list = list(id_deque)
            if len(id_deque_list) + num_new_id <= num_id_vocabulary:
                # The ID dictionary is not fully used, we can directly assign new ID words.
                new_ids = [len(id_deque_list) + _ for _ in range(num_new_id)]  # ID dictionary index (ID words)
            else:
                # The ID dictionary is fully used, we need to recycle some ID words.
                if len(id_deque_list) < num_id_vocabulary:
                    # There are still some empty slots in the ID dictionary,
                    # we can directly assign these clear_id_num_new_id new ID words.
                    clear_num_new_id = num_id_vocabulary - len(id_deque_list)
                    conflict_num_new_id = num_new_id - clear_num_new_id
                    new_ids = [len(id_deque_list) + _ for _ in range(clear_num_new_id)]
                else:
                    # There are no empty slots in the ID dictionary,
                    # we need to recycle conflict_num_new_id ID words.
                    conflict_num_new_id = num_new_id
                    new_ids = []
                # Recycled ID words:
                conflict_new_id = id_deque_list[:conflict_num_new_id]
                # As we need to recycle some ID words in conflict_new_id,
                # we need to remove the corresponding tracklets in the trajectory.
                for _ in range(len(trajectory)):
                    conflict_index = torch.zeros([len(trajectory[_]), ], dtype=torch.bool,
                                                 device=trajectory[_].ids.device)  # init
                    for _id in conflict_new_id:
                        conflict_index = conflict_index | (trajectory[_].ids == _id)
                    trajectory[_] = trajectory[_][~conflict_index]
                new_ids = new_ids + conflict_new_id  # assign the recycled ID words to "new_ids"

            # Update the corresponding mapping from ID words to ID labels (in tracker outputs):
            for _id in new_ids:
                ids_to_results[_id] = current_id
                current_id += 1
                id_deque.add(_id)

            # Insert the new_ids into the ids list:
            new_id_idx = 0
            ori_ids = ids
            ids = []
            for _ in ori_ids:
                if _ == -1:  # new id need to add:
                    ids.append(new_ids[new_id_idx])
                    new_id_idx += 1
                else:
                    ids.append(_)

        current[0].ids = torch.tensor(ids, dtype=torch.long, device=current[0].ids.device)
        trajectory_history_list = trajectory + current
        trajectory_history = deque(trajectory_history_list, maxlen=deque_max_length)
        assert len(ids) == len(set(ids)), f"ids is not unique: ids={ids}."
        return ids, trajectory_history, ids_to_results, current_id, id_deque, ~newborn_neg_filter
        # We will remove some illegal newborn targets in the outer function,
        # based on the "newborn_neg_filter" flags.

    def prepare(
            self,
            # The whole track_seqs, which contains the current detection results:
            track_seqs: list[list[Instances]],
            # The training augmentation parameters:
            traj_drop_ratio: float = 0.0,
            traj_switch_ratio: float = 0.0,
    ):
        assert len(track_seqs) == 1, f"SeqDecoder currently only support BS=1, but get BS={len(track_seqs)}."
        track_seq = track_seqs[0]  # for simplicity, we only use the first one

        # Here are some symbols we use:
        # N is the number of targets in the track_seq;
        # T is the temporal length of the track_seq;

        # All information is stored in a corresponding T-len list.
        all_ids = [_.id_words for _ in track_seq] if self.training else [_.ids for _ in track_seq]

        # all_features = [_.outputs for _ in track_seq]
        # all_boxes = [_.gt_boxes.detach().to(all_features[0].device) for _ in track_seq] if self.training \
        #     else [_.boxes.detach() for _ in track_seq]
        T = len(all_ids)
        # feature_dim = all_features[0].shape[-1]
        # box_dim = all_boxes[0].shape[-1]
        device = all_ids[0].device

        # Statistics of IDs that appear in the track_seq:
        all_ids_in_one_list = torch.cat(all_ids, dim=0).tolist()
        all_ids_set = set(all_ids_in_one_list)
        all_ids_set.discard(self.num_id_vocabulary)  # exclude the special ID token
        N = len(all_ids_set)

        # Build a mapping from ID to index, and index to ID:
        id_to_idx = {list(all_ids_set)[_]: list(range(N))[_] for _ in range(N)}
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        # Prepare the historical trajectory fields,
        # which should be in (N, T-1) shape.
        trajectory_ids_list, trajectory_features_list, trajectory_boxes_list, trajectory_times_list = [], [], [], []
        trajectory_masks_list = []
        idxs_temp = {}
        # Generate the historical trajectory fields:
        for t in range(T - 1):  # the historical trajectory only contains T-1 frames
            t_idxs = torch.tensor(
                [id_to_idx[_id.item()] for _id in all_ids[t]], dtype=torch.long, device=device
            )  # which index to use, for each object in current frame "t"
            idxs_temp[t] = t_idxs
            t_token_mask = torch.ones((N,), dtype=torch.bool, device=device)
            t_token_mask[t_idxs] = False  # in our code, False means the token is valid, True means invalid
            t_times = t * torch.ones((N,), dtype=torch.long, device=device)
            # Init fields:
            t_ids = -torch.ones((N,), dtype=torch.long, device=device)
            # t_features = torch.zeros((N, feature_dim), dtype=torch.float, device=device)
            # t_boxes = torch.zeros((N, box_dim), dtype=torch.float, device=device)
            # Fill fields:
            t_ids[t_idxs] = all_ids[t].to(device)
            # t_features[t_idxs] = all_features[t]
            # t_boxes[t_idxs] = all_boxes[t]
            # Append to the list:
            trajectory_ids_list.append(t_ids)
            # trajectory_features_list.append(t_features)
            # trajectory_boxes_list.append(t_boxes)
            trajectory_times_list.append(t_times)
            trajectory_masks_list.append(t_token_mask)
        # Stack the historical trajectory fields into tensors,
        # shape=(N, T-1, ...)
        # trajectory_features = torch.stack(trajectory_features_list, dim=1)
        # trajectory_boxes = torch.stack(trajectory_boxes_list, dim=1)
        # trajectory_times = torch.stack(trajectory_times_list, dim=1)
        trajectory_ids = torch.stack(trajectory_ids_list, dim=1)
        trajectory_masks = torch.stack(trajectory_masks_list, dim=1)

        # Prepare the current detection fields,
        # they have nearly the same attributes as historical trajectories.
        # We denote they as "unknown" because they need to be decoded.
        unknown_features_list, unknown_boxes_list, unknown_ids_list, unknown_times_list = [], [], [], []
        unknown_masks_list = []
        unknown_id_gts_list: list | None = [] if self.training else None

        if self.training:
            # During training, the last T-1 frames will be used to supervise the model,
            # so they are all "unknown".
            for t in range(1, T):
                N_t = len(all_ids[t])  # how many objects in this frame
                # Init fields:
                t_token_mask = torch.ones((N,), dtype=torch.bool, device=device)
                t_ids = -torch.ones((N,), dtype=torch.long, device=device)
                # t_features = torch.zeros((N, feature_dim), dtype=torch.float, device=device)
                # t_boxes = torch.zeros((N, box_dim), dtype=torch.float, device=device)
                # t_times = t * torch.ones((N,), dtype=torch.long, device=device)
                t_id_gts = -torch.ones((N,), dtype=torch.long, device=device)
                # Fill fields:
                if t in idxs_temp:
                    t_idxs = idxs_temp[t]  # this would be faster, but insignificant
                else:
                    t_idxs = torch.tensor(
                        [id_to_idx[_id.item()] for _id in all_ids[t]], dtype=torch.long, device=device
                    )
                t_token_mask[t_idxs] = False
                t_ids[t_idxs] = torch.tensor([self.num_id_vocabulary] * N_t, dtype=torch.long, device=device)
                # t_features[t_idxs] = all_features[t]
                # t_boxes[t_idxs] = all_boxes[t]
                t_id_gts[t_idxs] = track_seq[t].id_labels.to(device)
                # Append to the list:
                unknown_id_gts_list.append(t_id_gts)
                unknown_masks_list.append(t_token_mask)
                # unknown_times_list.append(t_times)
                unknown_ids_list.append(t_ids)
                # unknown_boxes_list.append(t_boxes)
                # unknown_features_list.append(t_features)
        else:
            # During inference, only the last frame will be used to decode.
            # And the number of objects in the last frame may be different from the previous frames,
            # so we need to redefine N_.
            N_ = len(all_ids[-1])
            # unknown_features_list.append(all_features[-1])
            # unknown_ids_list.append(all_ids[-1].to(device))
            # unknown_boxes_list.append(all_boxes[-1])
            unknown_times_list.append(torch.tensor([T - 1] * N_, dtype=torch.long, device=device))
            unknown_masks_list.append(torch.zeros((N_,), dtype=torch.bool, device=device))

        # Stack the current detection fields into tensors,
        # unknown_features = torch.stack(unknown_features_list, dim=1)
        # unknown_boxes = torch.stack(unknown_boxes_list, dim=1)
        unknown_ids = torch.stack(unknown_ids_list, dim=1)
        # unknown_times = torch.stack(unknown_times_list, dim=1)
        unknown_masks = torch.stack(unknown_masks_list, dim=1)
        unknown_id_gts = None if unknown_id_gts_list is None else torch.stack(unknown_id_gts_list, dim=1)

        # Training Augmentation:
        if self.training:
            N, T = trajectory_ids.shape[0], trajectory_ids.shape[1] + 1
            # Record which token is removed during this process:
            trajectory_remove_masks = torch.zeros((N, T - 1), dtype=torch.bool, device=device)

            # Trajectory Token Drop:
            for n in range(N):
                if random.random() < traj_drop_ratio:
                    traj_begin = random.randint(0, T - 1)
                    traj_max_t = T - 1 - traj_begin
                    traj_end = traj_begin + math.ceil(traj_max_t * random.random())
                    trajectory_remove_masks[n, traj_begin: traj_end] = True
            unknown_remove_masks = torch.cat([
                trajectory_remove_masks[:, 1:],
                torch.zeros((N, 1), dtype=torch.bool, device=device)
            ], dim=1)
            # Check if the trajectory augmentation process is legal.
            # Specifically, we need to ensure there is at least one object can be supervised,
            # or it may cause grad == None.
            # TODO: This legal check is just a simple implementation, it may not be rigorous.
            #       But it's enough for the current code.
            is_legal = (~(trajectory_masks | trajectory_remove_masks) & ~(
                    unknown_masks | unknown_remove_masks)).any().item()
            if is_legal:
                # We do not need to truly remove these tokens,
                # just set their to invalid tokens by masks.
                trajectory_masks = trajectory_masks | trajectory_remove_masks
                unknown_masks = unknown_masks | unknown_remove_masks
                # Also, we need to modify some ID ground-truths of "unknown" objects.
                # For example, a token first appears at t=3,
                # if I remove it at t=3, then I need to modify its ID GT at t=4 to the special token (newborn).
                for n in range(N):
                    line_traj_mask = trajectory_masks[n]
                    if line_traj_mask[0].item():
                        new_born_t = 0
                        for _ in line_traj_mask:
                            if _.item():
                                new_born_t += 1
                            else:
                                break
                        unknown_id_gts[n][:new_born_t] = self.num_id_vocabulary

            # Trajectory Token Switch (Swap):
            if traj_switch_ratio > 0.0:
                for t in range(0, T - 1):
                    switch_p = torch.ones((N,), dtype=torch.float, device=device) * traj_switch_ratio
                    switch_map = torch.bernoulli(switch_p)
                    switch_idxs = torch.nonzero(switch_map)  # objects to switch
                    switch_idxs = switch_idxs.reshape((switch_idxs.shape[0],))
                    if len(switch_idxs) == 1 and N > 1:
                        # Only one object can be switched, but we have more than one object.
                        # So we need to randomly select another object to switch.
                        switch_idxs = torch.as_tensor([switch_idxs[0].item(), random.randint(0, N - 1)],
                                                      dtype=torch.long, device=device)
                    if len(switch_idxs) > 1:
                        # Switch the trajectory features, boxes and masks:
                        shuffle_switch_idxs = switch_idxs[torch.randperm(len(switch_idxs)).to(device)]
                        # trajectory_features[switch_idxs, t, :] = trajectory_features[shuffle_switch_idxs, t, :]
                        # trajectory_boxes[switch_idxs, t, :] = trajectory_boxes[shuffle_switch_idxs, t, :]
                        trajectory_masks[switch_idxs, t] = trajectory_masks[shuffle_switch_idxs, t]
                    else:
                        continue  # no object to switch

        return [
            {
                #     "trajectory": {
                #         "features": trajectory_features,
                #         "boxes": trajectory_boxes,
                #         "ids": trajectory_ids,
                #         "times": trajectory_times,
                #         "masks": trajectory_masks,
                #         "pad_masks": None,
                #     },
                "unknown": {
                    # "features": unknown_features,
                    # "boxes": unknown_boxes,
                    "ids": unknown_ids,
                    # "times": unknown_times,
                    "masks": unknown_masks,
                    "id_gts": unknown_id_gts
                }
            }]

    def gt_id_deal(self, track_seqs: list[list[Instances]],
                   traj_drop_ratio: float = 0.0,
                   traj_switch_ratio: float = 0.0,
                   use_checkpoint: bool = False,
                   inference_ensemble: int = 0,
                   group_num=10
                   ):

        # if self.training or inference_ensemble == 0:
        #     format_seqs = self.prepare(
        #         track_seqs=track_seqs,
        #         traj_drop_ratio=0,
        #         traj_switch_ratio=0,
        #
        #     )
        #
        # else:
        #     assert inference_ensemble >= 2
        #     format_seqs = self.prepare(
        #         track_seqs=track_seqs,
        #         traj_drop_ratio=0,
        #         traj_switch_ratio=0,
        #     )

        emb_for_reid = {}

        for i, track_seq in enumerate(track_seqs[0]):
            if i == 0:
                for j, key_emb in enumerate(track_seq.ids):
                    emb_for_reid[int(key_emb)] = track_seq.pred_id_words[j, :].unsqueeze(0)
            elif i == 1:
                for j, key_emb in enumerate(track_seq.ids):
                    if int(key_emb) in emb_for_reid.keys():
                        emb_for_reid[int(key_emb)] = torch.cat(
                            [emb_for_reid[int(key_emb)], track_seq.pred_id_words[j, :].unsqueeze(0)], dim=0)
                    else:
                        emb_for_reid[int(key_emb)] = track_seq.pred_id_words[j, :].unsqueeze(0)
                id_pred_words = track_seq.pred_id_words
                id_gt_words = track_seq.ids
            else:
                id_pred_words = torch.cat([id_pred_words, track_seq.pred_id_words], dim=0)
                id_gt_words = torch.cat([id_gt_words, track_seq.ids], dim=0)
                for j, key_emb in enumerate(track_seq.ids):
                    # print(emb_for_reid.keys())
                    if int(key_emb) in emb_for_reid.keys():
                        # print(int(key_emb))
                        # print(emb_for_reid[int(key_emb)].shape)
                        # print(track_seq.pred_id_words[j, :].shape)
                        emb_for_reid[int(key_emb)] = torch.cat(
                            [emb_for_reid[int(key_emb)], track_seq.pred_id_words[j, :].unsqueeze(0)], dim=0)
                    else:
                        # print('track_seq.pred_id_words', track_seq.pred_id_words[j, :].shape)
                        # print('int(key_emb)', int(key_emb))

                        emb_for_reid[int(key_emb)] = track_seq.pred_id_words[j, :].unsqueeze(0)
                        # print('emb_for_reid[int(key_emb)]', emb_for_reid[int(key_emb)].shape)

        # id_gts = format_seqs[0]["unknown"]["id_gts"]
        #
        # if id_gts is not None:
        #     id_mask_flatten = einops.rearrange(format_seqs[0]["unknown"]["masks"], "n t -> (n t)")
        #     legal_id_gts = einops.rearrange(id_gts, "n t -> (n t)")[~id_mask_flatten]
        #     # id_gt_words = labels_to_one_hot(legal_id_gts, class_num=self.num_id_vocabulary + 1, device=self.device)
        #     id_gt_words = legal_id_gts
        # else:
        #     id_gt_words = None

        # id_gt_words = id_gt_words.repeat(group_num, 1)

        return id_pred_words[None, ...], id_gt_words[None, ...] if id_gt_words is not None else None, emb_for_reid

    def add_random_id_words_to_instances(self, instances: list[Instances]):
        # assert len(instances) == 1  # only for bs=1
        ids = torch.cat([instance.ids for instance in instances], dim=0)
        ids_unique = torch.unique(ids)
        # print(self.training_num_id)
        # print(ids_unique.shape)
        # print(ids)
        # exit()

        if len(ids_unique) > self.training_num_id:
            keep_index = torch.randperm(len(ids_unique))[:self.training_num_id]
            ids_unique = ids_unique[keep_index]
            pass
        id_words_unique = torch.randperm(n=self.num_id_vocabulary)[:len(ids_unique)]
        id_to_word = {
            i.item(): w.item() for i, w in zip(ids_unique, id_words_unique)
        }
        already_id_set = set()
        for t in range(len(instances)):
            id_words, id_labels = [], []
            for _ in range(len(instances[t])):
                i = instances[t].ids[_].item()
                if i in id_to_word:
                    id_words.append(id_to_word[i])
                else:  # handle the case that the number of objects exceeds the length of ID dictionary
                    id_words.append(-1)
                    id_labels.append(-1)
                    continue
                if i in already_id_set:
                    id_labels.append(id_to_word[i])
                else:
                    id_labels.append(self.num_id_vocabulary)
                    already_id_set.add(i)
            instances[t].id_words = torch.tensor(id_words, dtype=torch.long)
            instances[t].id_labels = torch.tensor(id_labels, dtype=torch.long)
            ins_keep_index = instances[t].id_words != -1
            instances[t] = instances[t][ins_keep_index]
        return

    def forward(self, frames, detr_checkpoint_frames: int | None = None, targets=None, max_gt_num=None,
                trajectory_output=None):

        if max_gt_num is None and targets is not None:
            num_gts = [len(t['labels']) for t in targets]
            max_gt_num = max(num_gts)

        if detr_checkpoint_frames is not None:
            # Checkpoint will only be used in the training stage.
            detr_outputs = None
            for batch_frames in batch_iterator(detr_checkpoint_frames, frames):
                batch_frames = batch_frames[0]
                if targets is None:
                    # _ = checkpoint(self.detr, batch_frames)
                    _ = self.detr(batch_frames, max_gt_num=max_gt_num, trajectory_inputs=trajectory_output)
                else:
                    if detr_outputs is None:
                        begin = 0
                    else:
                        begin = detr_outputs["outputs"].shape[0]
                    # _ = checkpoint(self.detr, batch_frames, targets[begin:begin+batch_frames.tensors.shape[0]], max_gt_num)
                    _ = self.detr(batch_frames, targets[begin:begin + batch_frames.tensors.shape[0]], max_gt_num,
                                  trajectory_inputs=trajectory_output)
                if detr_outputs is None:
                    detr_outputs = _
                else:
                    detr_outputs = combine_detr_outputs(detr_outputs, _)
        else:

            if targets is None:
                detr_outputs = self.detr(frames, trajectory_inputs=trajectory_output)
            else:
                detr_outputs = self.detr(frames, targets, max_gt_num=max_gt_num, trajectory_inputs=trajectory_output)

        return detr_outputs


def build(config: dict):
    return FastTrackTr(config=config)
