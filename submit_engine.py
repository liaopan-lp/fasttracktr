# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from data.seq_dataset import SeqDataset
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.utils import get_model
from utils.box_ops import box_cxcywh_to_xyxy
from collections import deque
from structures.instances import Instances
from structures.ordered_set import OrderedSet
from log.logger import Logger
from utils.utils import yaml_to_dict, is_distributed, distributed_rank, distributed_world_size
from models import build_rt_model
from models.utils import load_checkpoint
import time
from models.rt_detr_cross.tracker.multitracker import JDETracker
from models.rt_detr_cross.tracker.multitracker_plus import JDETrackerPlus
from models.rt_detr_cross.tracker.multitracker_plus_reid import JDETrackerPlusReID
from models.rt_detr_cross.tracker.args_for_multitracker import make_parser


def submit(config: dict, logger: Logger):
    """
    Submit a model for a specific dataset.
    :param config:
    :param logger:
    :return:
    """
    if config["INFERENCE_CONFIG_PATH"] is None:
        model_config = config
    else:
        model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_rt_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    if config["INFERENCE_GROUP"] is not None:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
                                          config["INFERENCE_SPLIT"],
                                          f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default",
                                          config["INFERENCE_SPLIT"],
                                          f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')

    # 需要调度整个 submit 流程
    submit_one_epoch(
        config=config,
        model=model,
        logger=logger,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=submit_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )

    logger.print(
        log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{os.path.join(submit_outputs_dir, 'tracker')}/.'")
    logger.save_log_to_file(
        log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{os.path.join(submit_outputs_dir, 'tracker')}/.'",
        filename="log.txt",
        mode="a"
    )

    return


@torch.no_grad()
def submit_one_epoch(config: dict, model: nn.Module,
                     logger: Logger, dataset: str, data_split: str,
                     outputs_dir: str, only_detr: bool = False):
    model.eval()

    all_seq_names = get_seq_names(data_root=config["DATA_ROOT"], dataset=dataset, data_split=data_split)
    seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                 if _ % distributed_world_size() == distributed_rank()]

    if len(seq_names) > 0:
        for seq in seq_names:
            submit_one_seq(
                model=model, dataset=dataset,
                seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, seq),
                only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
                outputs_dir=outputs_dir,
                det_thresh=config["DET_THRESH"],
                newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
            )
    else:  # fake submit, will not write any outputs.
        submit_one_seq(
            model=model, dataset=dataset,
            seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, all_seq_names[0]),
            only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            outputs_dir=outputs_dir,
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
            area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
            image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
            fake_submit=True,
            inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
        )

    if is_distributed():
        torch.distributed.barrier()

    return


@torch.no_grad()
def submit_one_seq(
        model: nn.Module, dataset: str, seq_dir: str, outputs_dir: str,
        only_detr: bool, max_temporal_length: int = 0,
        det_thresh: float = 0.2, newborn_thresh: float = 0.25, area_thresh: float = 100, id_thresh: float = 0.1,
        image_max_size: int = 1333,
        fake_submit: bool = False,
        inference_ensemble: int = 0,
        use_plus_tracker=True,
):
    os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
    seq_dataset = SeqDataset(seq_dir=seq_dir, dataset=dataset, width=image_max_size)
    seq_dataloader = DataLoader(seq_dataset, batch_size=1, num_workers=4, shuffle=False)
    # seq_name = seq_dir.split("/")[-1]
    seq_name = os.path.split(seq_dir)[-1]
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet()  # an ID deque for inference, the ID will be recycled if the dictionary is not enough.

    # Trajectory history:
    if only_detr:
        trajectory_history = None
    else:
        trajectory_history = deque(maxlen=max_temporal_length)
    if not use_plus_tracker:
        tracker = JDETracker(frame_rate=len(seq_dataloader))
    else:
        args = make_parser().parse_args()
        tracker = JDETrackerPlus(args, det_thresh=args.track_thresh, iou_threshold=args.iou_thresh,
                                 asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia,
                                 use_byte=args.use_byte)
        args_reid = make_parser().parse_args()
        args_reid.TCM_first_step = True
        args_reid.TCM_byte_step = True
        args_reid.TCM_first_step_weight = 1.0
        args_reid.TCM_byte_step_weight = 1.0
        args_reid.hybrid_sort_with_reid = True
        args_reid.with_fastreid = True
        args_reid.EG_weight_high_score = 4.0
        args_reid.EG_weight_low_score = 4.4

        if 'MOT17-05' in seq_name or 'MOT17-06' in seq_name:
            args_reid.track_buffer = 14
        elif 'MOT17-13' in seq_name or 'MOT17-14' in seq_name:
            args_reid.track_buffer = 25
        else:
            args_reid.track_buffer = 30

        if 'MOT17-01' in seq_name:
            args_reid.track_thresh = 0.65
        elif 'MOT17-06' in seq_name:
            args_reid.track_thresh = 0.65
        elif 'MOT17-12' in seq_name:
            args_reid.track_thresh = 0.7
        elif 'MOT17-14' in seq_name:
            args_reid.track_thresh = 0.67
        else:
            pass

        tracker_reid = JDETrackerPlusReID(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh,
                                          asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, args=args)

    if fake_submit:
        print(f"[Fake] Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")
    else:
        print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")

        # Loop over all frames in the sequence:
    trajectory_output = None
    results = []
    history_instances = Instances(image_size=(0, 0))
    for i, (image, ori_image) in enumerate(seq_dataloader):
        ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
        frame = tensor_list_to_nested_tensor([image[0]], is_training=False).to(device)

        start_time = time.time()
        detr_outputs = model(frames=frame, trajectory_output=trajectory_output)
        elapsed = time.time() - start_time
        # print(f'模型耗时： {elapsed * 1000} ms')
        # 写入txt保存耗时
        # with open(os.path.join(outputs_dir, "time.txt"), "a") as f:
        #     f.write(f"{seq_name.split('/')[-1]} {i + 1} {elapsed * 1000} ms\n")

        detr_logits = detr_outputs["pred_logits"][0]
        detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
        if not use_plus_tracker:
            detr_det_idxs = detr_scores > det_thresh  # filter by the detection threshold
            detr_det_logits = detr_logits[detr_det_idxs]
            detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
            detr_det_boxes = detr_outputs["pred_boxes"][0][detr_det_idxs]
            detr_det_outputs = detr_outputs["outputs"][0][detr_det_idxs]  # detr output embeddings
            area_legal_idxs = (detr_det_boxes[:, 2] * ori_w * detr_det_boxes[:,
                                                              3] * ori_h) > area_thresh  # filter by area
            detr_det_outputs = detr_det_outputs[area_legal_idxs]
            detr_det_boxes = detr_det_boxes[area_legal_idxs]
            detr_det_logits = detr_det_logits[area_legal_idxs]
            detr_pred_id_words = detr_outputs['id_words'][0][detr_det_idxs]
        else:
            detr_det_idxs = detr_scores > det_thresh  # filter by the detection threshold
            detr_det_boxes = detr_outputs["pred_boxes"][0]
            detr_det_logits = detr_scores
            detr_pred_id_words = detr_outputs['id_words'][0]

        value = detr_outputs['outputs']
        max_len = value.size(1)

        mask = torch.full((1, max_len), float('-inf'), dtype=value.dtype,
                          device=value.device)  # 初始化为 -inf
        mask[0, detr_det_idxs] = 0
        # pre_ref_point = detr_outputs['pred_boxes']
        if history_instances.is_empty():
            history_instances.history_embedding = value
        history_instances.history_output = value
        history_instances.mask = mask
        # history_instances.pre_ref_point = pre_ref_point

        query_pos_embed_history_encoder = detr_outputs['query_pos_embed_history_encoder']
        if history_instances.is_empty():
            history_instances.history_embedding = value
        history_instances.history_output = value
        history_instances.mask = mask
        history_instances.query_pos_embed = query_pos_embed_history_encoder

        trajectory_output = {"src": value, 'mask': mask}

        # De-normalize to target image size:
        box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])

        box_results = box_cxcywh_to_xyxy(boxes=box_results)
        if not use_plus_tracker:
            online_targets = tracker.update(box_results, detr_det_logits.cpu(), detr_pred_id_words.cpu())
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
            # save results
            results.append((i + 1, online_tlwhs, online_ids))
        else:
            start_time = time.time()
            online_targets = tracker.update(box_results, detr_scores.cpu(), detr_pred_id_words.cpu())
            # online_targets = tracker_reid.update(box_results, detr_scores.cpu(), detr_pred_id_words.cpu())
            elapsed = time.time() - start_time
            # print(f'关联耗时： {elapsed * 1000} ms') # 大概1.4ms，算的时候顶格算1.8ms
            online_xyxys = []
            online_ids = []
            for t in online_targets:
                # print(t)
                tid = int(t[-1])
                xyxy = t[:4]
                online_xyxys.append(xyxy)
                online_ids.append(tid)
            # print(online_xyxys)
            results.append((i + 1, online_xyxys, online_ids))

        # if show_image or save_dir is not None:
        #     online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=i,
        #                                   fps=1. / timer.average_time)
        # if show_image:
        #     cv2.imshow('online_im', online_im)
        # if save_dir is not None:
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        # frame_id += 1
        # save results
    result_file_path = os.path.join(outputs_dir, "tracker", f"{seq_name}.txt")
    if not use_plus_tracker:
        write_results(result_file_path, results, 'mot')
    else:
        write_plus_results(result_file_path, results, 'mot')

    return

    # if only_detr is False:
    #     if len(box_results) > get_model(model).num_id_vocabulary:
    #         print(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
    #               f"but get {len(box_results)} detections in seq {seq_name.split('/')[-1]} {i + 1}th frame.")

    # Decoding the current objects' IDs
    #     if only_detr is False:
    #         assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
    #                                             f"but get T={max_temporal_length - 1} history in Eval setting."
    #         current_tracks = Instances(image_size=(0, 0))
    #         current_tracks.boxes = detr_det_boxes
    #         current_tracks.outputs = detr_det_outputs
    #         current_tracks.ids = torch.tensor([get_model(model).num_id_vocabulary] * len(current_tracks),
    #                                           dtype=torch.long, device=current_tracks.outputs.device)
    #         current_tracks.confs = detr_det_logits.sigmoid()
    #         trajectory_history.append(current_tracks)
    #         if len(trajectory_history) == 1:  # first frame, do not need decoding:
    #             newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )  # filter by newborn
    #             trajectory_history[0] = trajectory_history[0][newborn_filter]
    #             box_results = box_results[newborn_filter.cpu()]
    #             ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
    #                                dtype=torch.long, device=current_tracks.outputs.device)
    #             trajectory_history[-1].ids = ids
    #             for _ in ids:
    #                 ids_to_results[_.item()] = current_id
    #                 current_id += 1
    #             id_results = []
    #             for _ in ids:
    #                 id_results.append(ids_to_results[_.item()])
    #                 id_deque.add(_.item())
    #             id_results = torch.tensor(id_results, dtype=torch.long)
    #         else:
    #             ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = get_model(model).update_id(
    #                 pred_id_words=detr_pred_id_words,
    #                 trajectory_history=trajectory_history,
    #                 num_id_vocabulary=get_model(model).num_id_vocabulary,
    #                 ids_to_results=ids_to_results,
    #                 current_id=current_id,
    #                 id_deque=id_deque,
    #                 id_thresh=id_thresh,
    #                 newborn_thresh=newborn_thresh,
    #                 inference_ensemble=inference_ensemble,
    #             )  # already update the trajectory history/ids_to_results/current_id/id_deque in this function
    #             id_results = []
    #             for _ in ids:
    #                 id_results.append(ids_to_results[_])
    #             id_results = torch.tensor(id_results, dtype=torch.long)
    #             if boxes_keep is not None:
    #                 box_results = box_results[boxes_keep.cpu()]
    #     else:  # only detr, ID is just +1 for each detection.
    #         id_results = torch.tensor([current_id + _ for _ in range(len(box_results))], dtype=torch.long)
    #         current_id += len(id_results)
    #
    #     # Output to tracker file:
    #     if fake_submit is False:
    #         # Write the outputs to the tracker file:
    #         result_file_path = os.path.join(outputs_dir, "tracker", f"{seq_name}.txt")
    #         with open(result_file_path, "a") as file:
    #             assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
    #                                                         f"but get len(IDs)={len(id_results)} and " \
    #                                                         f"len(Boxes)={len(box_results)}"
    #             for obj_id, box in zip(id_results, box_results):
    #                 obj_id = int(obj_id.item())
    #                 x1, y1, x2, y2 = box.tolist()
    #                 if dataset in ["DanceTrack", "MOT17", "SportsMOT", "MOT17_SPLIT", "MOT15", "MOT15_V2"]:
    #                     result_line = f"{i + 1}," \
    #                                   f"{obj_id}," \
    #                                   f"{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
    #                 else:
    #                     raise NotImplementedError(f"Do not know the outputs format of dataset '{dataset}'.")
    #                 file.write(result_line)
    # if fake_submit:
    #     print(f"[Fake] Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    # else:
    #     print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    print('save results to {}'.format(filename))


def write_plus_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w') as f:
        for frame_id, xyxys, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for xyxy, track_id in zip(xyxys, track_ids):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = xyxy
                # x2, y2 = x1 + w, y1 + h
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    print('save results to {}'.format(filename))


def get_seq_names(data_root: str, dataset: str, data_split: str):
    if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"]:
        dataset_dir = os.path.join(data_root, dataset, data_split)
        return sorted(os.listdir(dataset_dir))
    else:
        raise NotImplementedError(f"Do not support dataset '{dataset}' for eval dataset.")
