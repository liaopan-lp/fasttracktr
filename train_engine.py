# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import torch
import wandb
import torch.nn as nn
import torch.distributed

from einops import rearrange
from structures.instances import Instances
from torch.utils.data import DataLoader
from models import build_rt_model

from models.motip import MOTIP
from models.utils import save_checkpoint, load_checkpoint, load_detr_pretrain, get_model
from models.criterion import build as build_id_criterion
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, \
    combine_detr_outputs, detr_outputs_index_select, infos_to_detr_targets, batch_iterator, is_main_process, \
    combine_detr_outputs_bzs, init_first_output, \
    batch_iterator_for_bzs, get_detr_output_one_batch
from utils.nested_tensor import nested_tensor_index_select
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from log.logger import Logger, ProgressLogger
from log.log import Metrics, TPS
from eval_engine_cross import evaluate_one_epoch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import time
import copy
from models.criterion import build_multi_task_loss
import random
import re

def train(config: dict, logger: Logger):
    # Dataset:
    dataset_train = build_dataset(config=config)

    # Model
    model = build_rt_model(config=config)

    if config["DETR_PRETRAIN"] is not None:
        load_detr_pretrain(model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"])
        logger.print(f"Load DETR pretrain model from {config['DETR_PRETRAIN']}.")
    else:
        logger.print("No pre-trained detr used.")

    # For optimizer:
    param_groups = get_param_groups(model=model, config=config)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    # Criterion (Loss Function):
    id_criterion = build_id_criterion(config=config)
    multi_task_loss = build_multi_task_loss()
    # multi_task_loss = None
    device = torch.device(config["DEVICE"])
    if multi_task_loss is not None:
        multi_task_loss = multi_task_loss.to(device)

    # Scheduler:
    if config["SCHEDULER_TYPE"] == "MultiStep":
        scheduler = MultiStepLR(optimizer, milestones=config["SCHEDULER_MILESTONES"],
                                gamma=config["SCHEDULER_GAMMA"])
    else:
        raise RuntimeError(f"Do not support scheduler type {config['SCHEDULER_TYPE']}.")

    # Train States:
    train_states = {
        "start_epoch": 0,
        "global_iter": 0
    }

    # For resume:
    if config["RESUME_MODEL"] is not None:  # need to resume from checkpoint
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states if config["RESUME_STATES"] else None
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for i in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.print(f"Resume from model {config['RESUME_MODEL']}. "
                     f"Optimizer={config['RESUME_OPTIMIZER']}, Scheduler={config['RESUME_SCHEDULER']}, "
                     f"States={config['RESUME_STATES']}")
        logger.save_log_to_file(f"Resume from model {config['RESUME_MODEL']}. "
                                f"Optimizer={config['RESUME_OPTIMIZER']}, Scheduler={config['RESUME_SCHEDULER']}, "
                                f"States={config['RESUME_STATES']}", mode="a")

    # Distributed, every gpu will share the same parameters.
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()], broadcast_buffers=False,
                    find_unused_parameters=True)
        model._set_static_graph()

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        epoch_start_timestamp = TPS.timestamp()
        dataset_train.set_epoch(epoch)
        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader_train = build_dataloader(
            dataset=dataset_train,
            sampler=sampler_train,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"]
        )
        if is_distributed():
            sampler_train.set_epoch(epoch)

        # Train one epoch:
        train_metrics = train_one_epoch(
            config=config, model=model, logger=logger,
            dataloader=dataloader_train, id_criterion=id_criterion,
            optimizer=optimizer, epoch=epoch, states=train_states,
            clip_max_norm=config["CLIP_MAX_NORM"], detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_checkpoint_frames=config["DETR_CHECKPOINT_FRAMES"],
            lr_warmup_epochs=0 if "LR_WARMUP_EPOCHS" not in config else config["LR_WARMUP_EPOCHS"],
            group_num=1 if "GROUP_NUM" not in config else config["GROUP_NUM"],
            multi_task_loss=multi_task_loss

        )
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["learning_rate"].update(lr)
        train_metrics["learning_rate"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.print_metrics(
            metrics=train_metrics,
            prompt=f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ",
            fmt="{global_average:.4f}"
        )
        logger.save_metrics(
            metrics=train_metrics,
            prompt=f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ",
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_iter"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch"
        )

        # Save checkpoint.
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(model=model,
                            path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                            states=train_states,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            only_detr=config["TRAIN_STAGE"] == "only_detr",
                            )
            if config["INFERENCE_DATASET"] is not None:

                eval_model = copy.deepcopy(model)
                # eval_model = eval_model.half()

                if config["TRAIN_STAGE"] == "only_detr":
                    eval_metrics = evaluate_one_epoch(
                        config=config,
                        model=eval_model,
                        logger=logger,
                        dataset=config["INFERENCE_DATASET"],
                        data_split=config["INFERENCE_SPLIT"],
                        outputs_dir=os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                                                 "eval_during_train", config["INFERENCE_SPLIT"], f"epoch_{epoch}"),
                        only_detr=True
                    )
                else:
                    eval_metrics = evaluate_one_epoch(
                        config=config,
                        model=eval_model,
                        logger=logger,
                        dataset=config["INFERENCE_DATASET"],
                        data_split=config["INFERENCE_SPLIT"],
                        outputs_dir=os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                                                 "eval_during_train", config["INFERENCE_SPLIT"], f"epoch_{epoch}"),
                        only_detr=False
                    )
                eval_metrics.sync()
                logger.print_metrics(
                    metrics=eval_metrics,
                    prompt=f"[Epoch {epoch} Eval] ",
                    fmt="{global_average:.4f}"
                )
                logger.save_metrics(
                    metrics=eval_metrics,
                    prompt=f"[Epoch {epoch} Eval] ",
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_iter"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch"
                )

        # Next step.
        scheduler.step()

    return


def train_one_epoch(config: dict, model: MOTIP, logger: Logger,
                    dataloader: DataLoader, id_criterion: nn.Module,
                    optimizer: torch.optim,
                    epoch: int, states: dict, clip_max_norm: float, detr_num_train_frames: int,
                    detr_checkpoint_frames: int = 0, lr_warmup_epochs: int = 0, group_num: int = 1,
                    multi_task_loss: nn.Module = None):
    model.train()
    metrics = Metrics()  # save metrics
    memory_optimized_detr_criterion = config["MEMORY_OPTIMIZED_DETR_CRITERION"]
    checkpoint_detr_criterion = config["CHECKPOINT_DETR_CRITERION"]
    auto_memory_optimized_detr_criterion = config["AUTO_MEMORY_OPTIMIZED_DETR_CRITERION"]

    tps = TPS()  # save time per step

    device = torch.device(config["DEVICE"])

    # Check train stage:
    assert config["TRAIN_STAGE"] in ["only_detr", "only_decoder", "joint"], \
        f"Illegal train stage '{config['TRAIN_STAGE']}'."

    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    optimizer.zero_grad()  # init optim
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if epoch < lr_warmup_epochs:
            # Do lr warmup:
            lr_warmup(optimizer=optimizer, epoch=epoch, iteration=i,
                      orig_lr=config["LR"], warmup_epochs=lr_warmup_epochs, iter_per_epoch=len(dataloader))

        iter_start_timestamp = TPS.timestamp()

        # prepare some meta info
        num_gts = sum([len(info["boxes"]) for info in batch["infos"][0]])

        B, T = len(batch["images"]), len(batch["images"][0])
        history_instances = Instances(image_size=(0, 0))

        # Prepare frames for training:
        # print('detr_num_train_frames', detr_num_train_frames)
        frames = batch["nested_tensors"]  # (B, T, C, H, W) for tensors
        infos = batch["infos"]
        detr_targets = infos_to_detr_targets(infos=infos, device=device)
        random_frame_idxs = torch.randperm(T)
        argsort_random_frame_idx = torch.argsort(random_frame_idxs)
        argsort_random_frame_idx_repeat = torch.cat([argsort_random_frame_idx + b * T for b in range(B)])
        detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]
        # Prepare frames for training:
        detr_train_frames = nested_tensor_index_select(frames, dim=1, index=detr_train_frame_idxs)
        detr_no_grad_frames = nested_tensor_index_select(frames, dim=1, index=detr_no_grad_frame_idxs)
        detr_weight_dict = get_model(model).detr_criterion.weight_dict

        if B == 1:
            # (B, T) to (B*T):
            detr_train_frames.tensors = rearrange(detr_train_frames.tensors, "b t c h w -> (b t) c h w")
            detr_train_frames.mask = rearrange(detr_train_frames.mask, "b t h w -> (b t) h w")
            batch_frames_iterator = batch_iterator
            combine_detr_outputs_ = combine_detr_outputs
            detr_train_targets = [detr_targets[_] for _ in detr_train_frame_idxs.tolist()]
        else:
            # 多个batch size
            detr_train_frames.tensors = detr_train_frames.tensors.transpose(0, 1)
            detr_train_frames.mask = detr_train_frames.mask.transpose(0, 1)
            batch_frames_iterator = batch_iterator_for_bzs
            combine_detr_outputs_ = combine_detr_outputs_bzs
            detr_train_targets = []
            for detr_target in detr_targets:
                detr_train_targets.append([detr_target[_] for _ in detr_train_frame_idxs.tolist()])

        infos_for_train = []
        for info in infos:
            infos_for_train.append([info[_] for _ in detr_train_frame_idxs.tolist()])

        detr_train_frames = detr_train_frames.to(device)
        # detr_no_grad_frames.tensors = rearrange(detr_no_grad_frames.tensors, "b t c h w -> (b t) c h w")
        # detr_no_grad_frames.mask = rearrange(detr_no_grad_frames.mask, "b t h w -> (b t) h w")
        # detr_no_grad_frames = detr_no_grad_frames.to(device)

        # detr_train_targets = detr_targets

        trajectory_output = None

        # num_gts_tmp = [len(t['labels']) for t in detr_targets]
        # max_gt_num = max(num_gts_tmp)

        # DETR forward:
        detr_outputs = None
        target_num = 0
        frame_num = 0
        detr_loss = None

        for batch_frames in batch_frames_iterator(1, detr_train_frames):
            batch_frames = batch_frames[0]
            if B > 1:
                batch_frames.tensors = batch_frames.tensors.squeeze(0)
                batch_frames.mask = batch_frames.mask.squeeze(0)
                target = []
                for j in range(B):
                    target.append(detr_train_targets[j][frame_num])
            else:
                target = [detr_train_targets[frame_num]]

            frame_num += 1
            if detr_num_train_frames > 0:

                # detr_train_outputs_ = model(frames=batch_frames, targets=target,
                #                             max_gt_num=None,
                #                             trajectory_output=history_instances)
                # trajectory_output = {"src": history_output, 'mask': mask}
                detr_train_outputs_ = model(frames=batch_frames, targets=target,
                                            max_gt_num=None,
                                            trajectory_output=trajectory_output)
            else:
                detr_train_outputs_ = None

            # 每一帧要算一个match，用于在训练的时候生成mask，推理的时候
            # match_idxs = get_model(model).detr_criterion.matcher(outputs=detr_train_outputs_,
            #                                                      targets=target,
            #                                                      group_num=group_num, )

            detr_loss_dict, match_idxs = get_model(model).detr_criterion(outputs=detr_train_outputs_,
                                                                         targets=target)

            if detr_loss is None:
                # detr_loss = sum(
                #     detr_loss_dict[loss_key] * detr_weight_dict[loss_key] for loss_key in detr_loss_dict.keys() if
                #     loss_key in detr_weight_dict)
                detr_loss = sum(detr_loss_dict.values())
            else:
                # detr_loss += sum(
                #     detr_loss_dict[loss_key] * detr_weight_dict[loss_key] for loss_key in detr_loss_dict.keys() if
                #     loss_key in detr_weight_dict)
                detr_loss += sum(detr_loss_dict.values())

            # 下面是历史信息，可能之后会有别的处理
            # if len(match_idxs) > 1:
            #     outputs_match_idx = torch.cat([i[0].unsqueeze(0) for i in match_idxs], dim=0)
            # else:
            #     outputs_match_idx = match_idxs[0][0]
            # print(detr_train_outputs_)
            history_output = detr_train_outputs_[
                'outputs']  #.clone().detach()  # {key: value.clone().detach() for key, value in
            # detr_train_outputs_.items()}
            # masked_output = {}
            # for key, value in history_output.items():
            # mask = torch.zeros_like(value, dtype=torch.bool)
            # mask[outputs_match_idx] = 1
            # masked_value = value.masked_fill(~mask, float('-inf'))  # 使用 float('-inf') 遮盖不需要的张量
            # masked_output[key] = masked_value

            # 下面是transformer的mask写法，后面看看要不要encoder编码。

            max_len = history_output.size(1)  # max([value.size(0) for value in history_output.values()])
            mask = torch.full((B, max_len), float('-inf'), dtype=history_output.dtype,
                              device=history_output.device)  # 初始化为 -inf
            for a in range(B):
                outputs_match_idx = match_idxs[a][0]
                mask[a, outputs_match_idx] = 0
                # 随机取反步骤，取反的几率和 outputs_match_idx 的长度有关
                inversion_probability = len(outputs_match_idx) / max_len  # 取反概率与 outputs_match_idx 的长度成正比
                rand_vals = torch.rand(max_len, device=history_output.device)  # 生成一次随机数以加速计算
                outputs_mask = torch.zeros(max_len, dtype=torch.bool, device=history_output.device)
                # outputs_mask = torch.zeros(max_len, dtype=history_output.dtype, device=history_output.device)

                outputs_mask[outputs_match_idx] = True  # 标记 outputs_match_idx 中的元素
                specific_inversion_probabilities = torch.where(outputs_mask,
                                                               max(inversion_probability * 1.5, 0.2),
                                                               min(inversion_probability * 0.5, 0.2))  # 确定每个元素的取反概率
                # mask[a] ^= (rand_vals < specific_inversion_probabilities)  # 根据概率进行取反
                # 根据概率进行取反，保持可见的位置为 0
                # 对于需要遮掩的部分进行取反
                # Create a mask for inversion
                mask_inversion = rand_vals[a] < specific_inversion_probabilities
                mask[a, mask_inversion] = torch.where(mask[a, mask_inversion] == 0,
                                                      float('-inf'),
                                                      0)  # Invert based on the condition


            query_pos_embed_history_encoder = detr_train_outputs_['query_pos_embed_history_encoder']
            if history_instances.is_empty():
                history_instances.history_embedding = history_output
            history_instances.history_output = history_output
            history_instances.mask = mask
            history_instances.query_pos_embed = query_pos_embed_history_encoder

            history_output = detr_train_outputs_[
                'outputs']  # .clone().detach()  # {key: value.clone().detach() for key, value in
            trajectory_output = {"src": history_output, 'mask': mask}

            # trajectory_output = {"src": history_output, 'mask': mask, 'ref_point': pre_ref_point}

            if detr_outputs == None:
                if B == 1:
                    detr_outputs = detr_train_outputs_
                else:
                    detr_outputs = init_first_output(detr_train_outputs_)
            else:
                if B != 1:
                    detr_train_outputs_ = init_first_output(detr_train_outputs_)
                detr_outputs = combine_detr_outputs_(detr_outputs, detr_train_outputs_)

        # 计算ID loss
        if B == 1:
            # if checkpoint_detr_criterion:
            #     detr_loss_dict, match_idxs = checkpoint(
            #         get_model(model).detr_criterion,
            #         detr_outputs, detr_train_targets,
            #         use_reentrant=False
            #     )
            if checkpoint_detr_criterion:
                match_idxs = checkpoint(
                    get_model(model).detr_criterion.matcher,
                    detr_outputs, detr_train_targets
                )
            else:
                # detr_loss_dict, match_idxs = get_model(model).detr_criterion(outputs=detr_outputs, targets=detr_train_targets)
                outputs_without_aux = {k: v for k, v in detr_outputs.items() if
                                       k != 'aux_outputs' and k != 'enc_outputs'}

                match_idxs = get_model(model).detr_criterion.matcher(outputs_without_aux, detr_train_targets)

            # Generate field 'id_words' for instances:
            match_instances = generate_match_instances(
                match_idxs=match_idxs, infos=infos_for_train, detr_outputs=detr_outputs
            )
            # Generate field 'id_words' for instances:

            get_model(model).add_random_id_words_to_instances(instances=match_instances[0])

            pred_id_words, gt_id_words, emb_for_reid = get_model(model).gt_id_deal(
                track_seqs=match_instances,
                traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
                group_num=group_num
            )

            id_loss = id_criterion(pred_id_words, gt_id_words, emb_for_reid)

            if 'aux_outputs' in detr_outputs:
                id_loss_ = 0
                for x, aux_outputs in enumerate(detr_outputs['aux_outputs']):
                    match_idxs = get_model(model).detr_criterion.matcher(outputs=aux_outputs,
                                                                         targets=detr_train_targets)
                    match_instances = generate_match_instances(
                        match_idxs=match_idxs, infos=infos_for_train, detr_outputs=aux_outputs
                    )
                    get_model(model).add_random_id_words_to_instances(instances=match_instances[0])
                    pred_id_words, gt_id_words, emb_for_reid = get_model(model).gt_id_deal(
                        track_seqs=match_instances,
                        traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                        traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                        use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
                        group_num=group_num
                    )
                    id_loss_ += id_criterion(pred_id_words, gt_id_words, emb_for_reid)
                id_loss_ = id_loss_ / (x+1)
                id_loss += id_loss_

            # Calculate the overall loss for barkward processing:

            # detr_loss = sum(
            #     detr_loss_dict[loss_key] * detr_weight_dict[loss_key] for loss_key in detr_loss_dict.keys() if loss_key in detr_weight_dict)
        else:
            id_loss = 0
            for k in range(B):
                detr_output = get_detr_output_one_batch(detr_outputs, k)

                # detr_loss_dict, match_idxs = get_model(model).detr_criterion(outputs=detr_output,
                #                                                              targets=detr_train_targets[k])

                outputs_without_aux = {k: v for k, v in detr_output.items() if
                                       k != 'aux_outputs' and k != 'enc_outputs'}

                match_idxs = get_model(model).detr_criterion.matcher(outputs=outputs_without_aux,
                                                                     targets=detr_train_targets[k])

                # MOTIP processing:
                match_instances = generate_match_instances(
                    match_idxs=match_idxs, infos=[infos_for_train[k]], detr_outputs=detr_output
                )
                # Generate field 'id_words' for instances:

                get_model(model).add_random_id_words_to_instances(instances=match_instances[0])


                pred_id_words, gt_id_words, emb_for_reid = get_model(model).gt_id_deal(
                    track_seqs=match_instances,
                    traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                    traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                    use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
                    group_num=group_num
                )

                if k == 0:
                    id_loss = id_criterion(pred_id_words, gt_id_words, emb_for_reid)
                    # Calculate the overall loss for barkward processing:
                    # detr_weight_dict = get_model(model).detr_criterion.weight_dict
                    # detr_loss = sum(
                    #     detr_loss_dict[loss_key] * detr_weight_dict[loss_key] for loss_key in detr_loss_dict.keys() if
                    #     loss_key in detr_weight_dict)
                else:
                    id_loss += id_criterion(pred_id_words, gt_id_words, emb_for_reid)
                    # detr_weight_dict = get_model(model).detr_criterion.weight_dict
                    #
                    # detr_loss += sum(
                    #     detr_loss_dict[loss_key] * detr_weight_dict[loss_key] for loss_key in detr_loss_dict.keys() if
                    #     loss_key in detr_weight_dict)

                if 'aux_outputs' in detr_output:
                    id_loss_ = None
                    for x, aux_outputs in enumerate(detr_output['aux_outputs']):
                        match_idxs = get_model(model).detr_criterion.matcher(outputs=aux_outputs,
                                                                             targets=detr_train_targets[k])
                        match_instances = generate_match_instances(
                            match_idxs=match_idxs, infos=[infos_for_train[k]], detr_outputs=aux_outputs
                        )
                        get_model(model).add_random_id_words_to_instances(instances=match_instances[0])
                        pred_id_words, gt_id_words, emb_for_reid = get_model(model).gt_id_deal(
                            track_seqs=match_instances,
                            traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                            traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                            use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
                            group_num=group_num
                        )
                        if id_loss_ is None:
                            id_loss_ = id_criterion(pred_id_words, gt_id_words, emb_for_reid)
                        else:
                            id_loss_ += id_criterion(pred_id_words, gt_id_words, emb_for_reid)
                    id_loss_ = id_loss_ / x
                    id_loss += id_loss_
        if get_model(model).detr.multi_task_loss is None or id_loss == 0:
            loss = detr_loss  #+ id_loss * id_criterion.weight
        else:
            loss = get_model(model).detr.multi_task_loss(detr_loss, id_loss)
        # loss = detr_loss + id_loss * id_criterion.weight
        # loss /= T

        # Backward the loss:
        loss /= config["ACCUMULATE_STEPS"]

        loss.backward()
        # torch.cuda.empty_cache()

        # Add metrics to Log:
        metrics["overall_loss"].update(loss.item())
        metrics["overall_detr_loss"].update(detr_loss.item())
        metrics["bbox_l1"].update(detr_loss_dict["loss_bbox"].item())
        metrics["bbox_giou"].update(detr_loss_dict["loss_giou"].item())
        metrics["cls_loss"].update(detr_loss_dict["loss_ce"].item())
        if config["TRAIN_STAGE"] != "only_detr" and id_loss != 0:  # log about id branch is also need to be written:
            metrics["overall_id_loss"].update(id_loss.item() * id_criterion.weight)
            metrics["id_loss"].update(id_loss.item())

        if get_model(model).detr.multi_task_loss is not None:
            metrics["log_sigma_det"].update(get_model(model).detr.multi_task_loss.log_sigma_det.item())
            metrics["log_sigma_reid"].update(get_model(model).detr.multi_task_loss.log_sigma_reid.item())

        # Parameters update:
        if (i + 1) % config["ACCUMULATE_STEPS"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            if clip_max_norm > 0:
                detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, clip_max_norm)
                other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, clip_max_norm)
                metrics["detr_grad_norm"].update(detr_grad_norm.item())
                metrics["other_grad_norm"].update(other_grad_norm.item())
            else:
                pass

        iter_end_timestamp = TPS.timestamp()
        tps.update(iter_end_timestamp - iter_start_timestamp)
        eta = tps.eta(total_steps=len(dataloader), current_steps=i)

        if (i % config["OUTPUTS_PER_STEP"] == 0) or (i == len(dataloader) - 1):
            metrics["learning_rate"].clear()
            metrics["learning_rate"].update(optimizer.state_dict()["param_groups"][-1]["lr"])
            metrics.sync()
            logger.print_metrics(
                metrics=metrics,
                prompt=f"[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] "
            )
            logger.save_metrics(
                metrics=metrics,
                prompt=f"[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] ",
                global_step=states["global_iter"],
            )

        states["global_iter"] += 1

    states["start_epoch"] += 1

    return metrics


def generate_match_instances(match_idxs, infos, detr_outputs):
    match_instances = []
    B, T = len(infos), len(infos[0])
    for b in range(B):
        match_instances.append([])
        for t in range(T):
            flat_idx = b * T + t
            output_idxs, info_idxs = match_idxs[flat_idx]
            instances = Instances(image_size=(0, 0))
            instances.ids = infos[b][t]["ids"][info_idxs]
            instances.gt_boxes = infos[b][t]["boxes"][info_idxs]
            instances.pred_boxes = detr_outputs["pred_boxes"][flat_idx][output_idxs]
            # instances.outputs = detr_outputs["outputs"][flat_idx][output_idxs]
            instances.pred_id_words = detr_outputs["id_words"][flat_idx][output_idxs]
            # print('pred_id_words len',len(instances.pred_id_words))
            # instances.outputs_ids = detr_outputs["pred_ids"][flat_idx][output_idxs]

            match_instances[b].append(instances)
    return match_instances


# def get_param_groups(model: nn.Module, config) -> list[dict]:
#     def match_names(name, key_names):
#         for key in key_names:
#             if key in name:
#                 return True
#         return False
#
#     # keywords
#     backbone_names = config["LR_BACKBONE_NAMES"]
#     linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
#     dictionary_names = [] if "LR_DICTIONARY_NAMES" not in config else config["LR_DICTIONARY_NAMES"]
#     _dictionary_scale = 1.0 if "LR_DICTIONARY_SCALE" not in config else config["LR_DICTIONARY_SCALE"]
#     param_groups = [
#         {
#             "params": [p for n, p in model.named_parameters() if match_names(n, backbone_names) and p.requires_grad],
#             "lr_scale": config["LR_BACKBONE_SCALE"],
#             "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if match_names(n, linear_proj_names) and p.requires_grad],
#             "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
#             "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if match_names(n, dictionary_names) and p.requires_grad],
#             "lr_scale": _dictionary_scale,
#             "lr": config["LR"] * _dictionary_scale
#         },
#         {
#             "params": [p for n, p in model.named_parameters()
#                        if not match_names(n, backbone_names)
#                        and not match_names(n, linear_proj_names)
#                        and not match_names(n, dictionary_names)
#                        and p.requires_grad],
#         }
#     ]
#     return param_groups

import re

def get_param_groups(model: nn.Module, config) -> list[dict]:
    def match_names(name, key_names):
        """检查参数名称是否包含任何关键字"""
        for key in key_names:
            if key in name:
                return True
        return False

    def matches_weight_decay(name, regex_patterns):
        """检查参数名称是否匹配任何weight_decay正则表达式"""
        for pattern in regex_patterns:
            if pattern.match(name):
                return True
        return False

    # 提取配置参数，使用get以避免KeyError
    backbone_names = config.get("LR_BACKBONE_NAMES", [])
    linear_proj_names = config.get("LR_LINEAR_PROJ_NAMES", [])
    dictionary_names = config.get("LR_DICTIONARY_NAMES", [])
    weight_decay_dictionary_names = config.get("WEIGHT_DECAY_NAMES", [])  # 确认键名是否正确
    _dictionary_scale = config.get("LR_DICTIONARY_SCALE", 1.0)
    default_weight_decay = config.get("WEIGHT_DECAY", 0.01)  # 设置默认的weight_decay值

    # 预编译正则表达式模式以提高效率
    weight_decay_patterns = [re.compile(pattern) for pattern in weight_decay_dictionary_names]

    # 初始化参数组列表
    param_groups = []

    # 辅助函数：根据参数名称确定weight_decay值
    def get_weight_decay(name):
        if matches_weight_decay(name, weight_decay_patterns):
            return 0.0
        return default_weight_decay

    # 定义各个参数组的处理逻辑
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数

        # 确定学习率和学习率缩放
        if match_names(name, backbone_names):
            lr_scale = config["LR_BACKBONE_SCALE"]
            lr = config["LR"] * lr_scale
            group_type = "backbone"
        elif match_names(name, linear_proj_names):
            lr_scale = config["LR_LINEAR_PROJ_SCALE"]
            lr = config["LR"] * lr_scale
            group_type = "linear_proj"
        elif match_names(name, dictionary_names):
            lr_scale = _dictionary_scale
            lr = config["LR"] * lr_scale
            group_type = "dictionary"
        else:
            lr_scale = 1.0
            lr = config["LR"]
            group_type = "others"

        # 确定weight_decay
        weight_decay = get_weight_decay(name)

        # 查找是否已经存在一个相同设置的参数组
        group_found = False
        for group in param_groups:
            if (group.get("group_type") == group_type and
                group.get("lr_scale") == lr_scale and
                group.get("weight_decay") == weight_decay):
                group["params"].append(param)
                group_found = True
                break

        # 如果没有找到匹配的参数组，则创建一个新的
        if not group_found:
            param_group = {
                "params": [param],
                "group_type": group_type,  # 仅用于内部逻辑，不会影响优化器
                "lr_scale": lr_scale,
                "lr": lr,
                "weight_decay": weight_decay
            }
            param_groups.append(param_group)

    # 移除内部使用的"group_type"键
    for group in param_groups:
        group.pop("group_type", None)

    return param_groups




def lr_warmup(optimizer, epoch: int, iteration: int, orig_lr: float, warmup_epochs: int, iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * iter_per_epoch
    current_lr_ratio = (epoch * iter_per_epoch + iteration + 1) / total_warmup_iters
    current_lr = orig_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return
