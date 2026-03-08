"""
dataset.py
----------
DataLoader 工厂函数。

修复说明：
  - 使用 cfg["clip_model_path"]（本地路径）加载 CLIP preprocess，
    避免重新从网络下载 890M 权重文件。
"""

import clip
import torch
from torch.utils.data import DataLoader

from .composition_dataset import CompositionDataset


def get_dataloader(cfg: dict, split: str) -> tuple:
    """
    Parameters
    ----------
    cfg   : 配置字典
    split : "train" | "val" | "test"

    Returns
    -------
    (DataLoader, CompositionDataset)
    """
    # ── 用本地路径加载 CLIP preprocess，避免联网下载 ──
    # clip.load 传入本地 .pt 路径时，只加载权重不下载；
    # preprocess 的逻辑与模型名称无关，只取决于 image_size（ViT-L/14 固定 224）
    local_path = cfg.get("clip_model_path", None)
    load_target = local_path if local_path else cfg.get("clip_model", "ViT-L/14")

    _, preprocess = clip.load(
        load_target,
        device="cpu",
        jit=False,
    )

    dataset = CompositionDataset(
        data_dir  = cfg["data_dir"],
        split     = split,
        transform = preprocess,
    )

    is_train = (split == "train")
    loader   = DataLoader(
        dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = is_train,
        num_workers = cfg.get("num_workers", 4),
        pin_memory  = True,
        drop_last   = is_train,
    )

    return loader, dataset