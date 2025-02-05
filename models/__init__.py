# Copyright (c) RuopengGao. All Rights Reserved.
import torch

from .motip import build as build_motl
from .rtmot import build as build_rtmot
from utils.utils import distributed_rank
from .fasttracktr import build as build_rtmot_cross


def build_model(config: dict):
    # model = build_motl(config=config)
    model = build_rtmot(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model

def build_rt_model(config: dict):
    model = build_rtmot_cross(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model