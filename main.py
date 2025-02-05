# Copyright (c) LiaoPan. All Rights Reserved.
import os
import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed
from log.logger import Logger, parser_to_dict
from configs.utils import update_config, load_super_config
from train_engine_corss import train
from eval_engine_cross import evaluate
from submit_engine_cross import submit


def parse_option():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    Every option in this parser should appear in .yaml config file (like ./configs/resnet18_mnist.yaml),
    except --config-path.

    Returns:
        A parser.

    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Git:
    parser.add_argument("--git-version", default='', type=str)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", default='train', type=str, help="Running mode.")

    # About model setting:
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--detr-num-queries", type=int)
    parser.add_argument("--detr-pretrain", default=None, type=str)  # DETR pretrain
    # parser.add_argument("--num-seq-decoder-layers", type=int)
    parser.add_argument("--seq-hidden-dim", type=int)
    parser.add_argument("--seq-dim-feedforward", type=int)
    parser.add_argument("--id-decoder-layers", type=int)
    parser.add_argument("--num-id-vocabulary", type=int)

    # Config file.
    parser.add_argument("--config-path", type=str, default='./configs/r50_rt_detr_rtmot_dancetrack.yaml',
                        help="Config file path.")
    parser.add_argument("--super-config-path", type=str)

    # About system.
    parser.add_argument("--device", type=str, help="Device.")
    parser.add_argument("--num-cpu-per-gpu", type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")
    parser.add_argument("--data-root", default=r'H:\data', type=str)
    parser.add_argument("--sample-lengths", nargs="*", type=int)
    parser.add_argument("--sample-intervals", nargs="*", type=int)
    parser.add_argument("--max-temporal-length", type=int)
    parser.add_argument("--aug-random-shift-max-ratio", type=float)
    parser.add_argument("--dataset-weights", nargs="*", type=float)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--dataset-splits", nargs="*", type=str)

    # About outputs.
    parser.add_argument("--use-wandb", default=False, type=str)
    parser.add_argument("--outputs-dir", default='./output', type=str, help="Outputs dir.")
    parser.add_argument("--exp-name", type=str, help="Exp name.")
    parser.add_argument("--exp-group", type=str, help="Exp group, for wandb.")
    parser.add_argument("--save-checkpoint-per-epoch", type=int)

    # About train setting:
    parser.add_argument("--resume-model", type=str, help="Resume training model path.")
    parser.add_argument("--resume-optimizer", type=str)
    parser.add_argument("--resume-scheduler", type=str)
    parser.add_argument("--detr-num-train-frames", type=int)
    parser.add_argument("--accumulate-steps", type=int)
    parser.add_argument("--detr-checkpoint-frames", default=1, type=int)
    parser.add_argument("--seq-decoder-checkpoint", type=str)
    parser.add_argument("--training-num-id", type=int)
    parser.add_argument("--memory-optimized-detr-criterion", type=str)
    parser.add_argument("--auto-memory-optimized-detr-criterion", type=str)
    parser.add_argument("--checkpoint-detr-criterion", type=str)
    # Training augmentation parameters:
    parser.add_argument("--traj-drop-ratio", type=float)
    parser.add_argument("--traj-switch-ratio", type=float)
    parser.add_argument("--group-num", default=1, type=int)

    # About evaluation and submit:
    parser.add_argument("--inference-model", default=r".\output\checkpoint_0.pth", type=str)
    # parser.add_argument("--inference-config-path", default='./configs/r50_rt_detr_rtmot_dancetrack.yaml', type=str)
    parser.add_argument("--inference-config-path", default='./configs/r50_rt_detr_rtmot_dancetrack.yaml', type=str)
    parser.add_argument("--inference-group", type=str)
    parser.add_argument("--inference-split", type=str)
    parser.add_argument("--inference-dataset", type=str)
    parser.add_argument("--inference-max-size", type=int)

    # Distributed.
    parser.add_argument("--use-distributed", default=False, type=str, help="Whether use distributed mode.")

    # Hyperparams.
    parser.add_argument("--epochs", default=35, type=int)
    parser.add_argument("--scheduler-milestones", type=int, nargs="*")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-warmup-epochs", type=int)
    parser.add_argument("--id-thresh", type=float)
    parser.add_argument("--det-thresh", type=float)
    parser.add_argument("--newborn-thresh", type=float)
    parser.add_argument("--area-thresh", type=int)
    parser.add_argument("--detr-cls-loss-coef", type=float)

    return parser.parse_args()


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]  # setting available gpus, like: "0,1,2,3"

    if config["USE_DISTRIBUTED"]:
        # https://i.steer.space/blog/2021/01/pytorch-dist-nccl-backend-allgather-stuck
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    # You can only set the `--exp-name` in runtime option,
    # if `--outputs-dir` is None, it will be set to `./outputs/[exp_name]/`.
    if config["OUTPUTS_DIR"] is None:
        config["OUTPUTS_DIR"] = os.path.join("./outputs", config["EXP_NAME"])

    if config["MODE"] == "train":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
    elif config["MODE"] == "eval":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                               config["INFERENCE_GROUP"] if config["INFERENCE_GROUP"] is not None else "default",
                               config["INFERENCE_SPLIT"])
    elif config["MODE"] == "submit":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                               config["INFERENCE_GROUP"] if config["INFERENCE_GROUP"] is not None else "default",
                               config["INFERENCE_SPLIT"])
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}' yet.")

    logger = Logger(
        logdir=log_dir,
        use_tensorboard=config["USE_TENSORBOARD"],
        use_wandb=config["USE_WANDB"],
        only_main=True,
        config=config
    )
    # Log runtime config.
    if is_main_process():
        logger.print_config(config=config, prompt="Runtime Configs: ")
        logger.save_config(config=config, filename="config.yaml")
        # logger.show(log=config, prompt="Main configs: ")
        # logger.write(config, "config.yaml")

    # set seed
    set_seed(config["SEED"])
    # Set num of CPUs
    if "NUM_CPU_PER_GPU" in config and config["NUM_CPU_PER_GPU"] is not None:
        torch.set_num_threads(config["NUM_CPU_PER_GPU"])

    return logger


if __name__ == '__main__':
    opt = parse_option()  # runtime options, a subset of .yaml config file (dict).
    cfg = yaml_to_dict(opt.config_path)  # configs from .yaml file, path is set by runtime options.

    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    logger = main(config=update_config(config=cfg, option=opt))

    if cfg["MODE"] == "train":
        train(config=cfg, logger=logger)
    elif cfg["MODE"] == "eval":
        evaluate(config=cfg, logger=logger)
    elif cfg["MODE"] == "submit":
        submit(config=cfg, logger=logger)
