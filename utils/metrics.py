"""
metrics.py
----------
CZSL 指标计算工具函数。
"""

import numpy as np
from typing import Tuple


def compute_auc(
    seen_accs  : np.ndarray,
    unseen_accs: np.ndarray,
) -> float:
    """
    计算 Seen-Unseen 曲线下的 AUC。

    通过对 seen 数组排序后使用梯形积分近似面积。
    由于改变 bias 会导致 seen ↓ unseen ↑，
    曲线在 [0,1]×[0,1] 空间内。

    Parameters
    ----------
    seen_accs   : 不同 bias 下的 seen 准确率数组
    unseen_accs : 不同 bias 下的 unseen 准确率数组

    Returns
    -------
    auc : float，范围 [0, 100]（百分比形式）
    """
    # 按 seen acc 排序（确保积分方向正确）
    order       = np.argsort(seen_accs)
    seen_sorted = seen_accs[order]
    unseen_sorted = unseen_accs[order]

    auc = float(np.trapz(unseen_sorted, seen_sorted))
    if auc < 0:
        auc = -auc
    return auc * 100.0


def compute_harmonic_mean(seen: float, unseen: float) -> float:
    """计算调和平均数"""
    if seen + unseen < 1e-8:
        return 0.0
    return 2 * seen * unseen / (seen + unseen)


def format_metrics(metrics: dict) -> str:
    """将指标字典格式化为易读字符串"""
    return (
        f"S={metrics.get('S', 0):.2f}  "
        f"U={metrics.get('U', 0):.2f}  "
        f"HM={metrics.get('HM', 0):.2f}  "
        f"AUC={metrics.get('AUC', 0):.2f}"
    )