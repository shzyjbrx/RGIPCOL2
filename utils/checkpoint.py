"""
checkpoint.py
-------------
模型检查点的保存与加载。
"""

import os
from typing import Optional

import torch


def save_checkpoint(
    model    ,
    optimizer,
    epoch    : int,
    metrics  : dict,
    path     : str,
):
    """
    保存模型检查点。

    只保存可训练参数（prefix_vectors 和 RGCN 参数），
    不保存冻结的 CLIP 参数（节省磁盘空间）。
    """
    # 只保存 requires_grad=True 的参数
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(k.startswith(prefix) for prefix in ["prompt_learner", "rgcn"])
    }

    checkpoint = {
        "epoch"         : epoch,
        "metrics"       : metrics,
        "model_state"   : trainable_state,
        "optimizer_state": optimizer.state_dict(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model    ,
    path     : str,
    optimizer  = None,
    device   : torch.device = None,
) -> dict:
    """
    从检查点文件加载可训练参数。

    Returns
    -------
    checkpoint 字典（包含 epoch, metrics 等信息）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Checkpoint] 文件不存在：{path}")

    map_location = device if device is not None else "cpu"
    checkpoint   = torch.load(path, map_location=map_location)

    # 只加载可训练参数（忽略 CLIP 冻结参数的 key mismatch）
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state"], strict=False
    )
    if missing:
        print(f"[Checkpoint] 缺少的 key（可能是冻结层，正常）：{missing[:5]}...")
    if unexpected:
        print(f"[Checkpoint] 意外的 key：{unexpected[:5]}...")

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch   = checkpoint.get("epoch",   -1)
    metrics = checkpoint.get("metrics", {})
    print(f"[Checkpoint] 加载完成：epoch={epoch}, metrics={metrics}")

    return checkpoint