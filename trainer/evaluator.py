"""
evaluator.py
------------
CZSL 标准评估器。

计算四个指标（闭/开世界均支持）：
  - S   : Best Seen Accuracy   （bias → -∞）
  - U   : Best Unseen Accuracy  （bias → +∞）
  - HM  : Best Harmonic Mean    （扫描 bias 找最优 HM）
  - AUC : Area Under Seen-Unseen Accuracy Curve

评估流程：
  1. 对所有图像，在候选 pair 集合上计算相似度分数
  2. 为 unseen pair 批量添加 bias（从 -∞ 到 +∞ 扫描）
  3. 计算每个 bias 下的 seen_acc 和 unseen_acc
  4. 计算 S / U / HM / AUC
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from trainer.feasibility import FeasibilityCalibrator


class Evaluator:
    """
    Parameters
    ----------
    model   : RGIPCOL 实例
    dataset : val 或 test 的 CompositionDataset
    cfg     : 配置字典
    device  : 设备
    """

    def __init__(self, model, dataset, cfg: dict, device: torch.device):
        self.model   = model
        self.dataset = dataset
        self.cfg     = cfg
        self.device  = device

        # ── 构建 seen / unseen 集合的索引（在 closed_pairs 中的位置） ──
        self.closed_pairs = dataset.closed_pairs        # seen ∪ unseen
        self.seen_set     = set(dataset.seen_pairs)
        self.unseen_set   = set(dataset.unseen_pairs)

        self.seen_indices   = [
            i for i, p in enumerate(self.closed_pairs) if p in self.seen_set
        ]
        self.unseen_indices = [
            i for i, p in enumerate(self.closed_pairs) if p in self.unseen_set
        ]

        # # ── bias 扫描范围 ──
        # bias_step = cfg.get("bias_step", 0.1)
        # self.biases = np.arange(-3.0, 3.0 + bias_step, bias_step)
        
        # ── 极细粒度的余弦相似度 Bias 扫描范围 ──
        # 由于不使用 logit_scale，分数在 [-1, 1] 内。0.001 的步长能精准捕捉曲线的交叉点。
        bias_step = 0.001 
        self.biases = np.arange(-1.0, 1.0 + bias_step, bias_step)

    # ────────────────────────────
    # 闭合世界评估
    # ────────────────────────────

    def evaluate_closed_world(self, dataloader) -> Dict[str, float]:
        """
        Returns
        -------
        dict with keys: S, U, HM, AUC
        """
        # ── 收集所有样本的分数和真实标签 ──
        all_scores, all_gt_attr, all_gt_obj = self._collect_scores(
            dataloader, self.closed_pairs
        )
        # all_scores : (N, P)  — N=样本数，P=候选pair数
        # all_gt_attr, all_gt_obj : (N,)

        metrics = self._compute_metrics(
            all_scores  = all_scores,
            gt_attr     = all_gt_attr,
            gt_obj      = all_gt_obj,
            pairs       = self.closed_pairs,
            seen_idx    = self.seen_indices,
            unseen_idx  = self.unseen_indices,
        )
        return metrics

    # ────────────────────────────
    # 开放世界评估
    # ────────────────────────────

    def evaluate_open_world(
        self,
        dataloader,
        feasibility_calibrator: Optional[FeasibilityCalibrator] = None,
    ) -> Dict[str, float]:
        """
        开放世界：目标集为全部 attr×obj 对。
        先用可行性校准过滤不可行对，再计算指标。

        Returns
        -------
        dict with keys: S, U, HM, AUC
        """
        all_pairs   = self.dataset.all_pairs   # 全部笛卡尔积

        # seen / unseen 在全部 pair 中的索引
        seen_set   = self.seen_set
        unseen_set = self.unseen_set
        seen_idx_ow   = [i for i, p in enumerate(all_pairs) if p in seen_set]
        unseen_idx_ow = [i for i, p in enumerate(all_pairs) if p in unseen_set]

        # ── 收集分数 ──
        all_scores, all_gt_attr, all_gt_obj = self._collect_scores(
            dataloader, all_pairs
        )

        # ── 可行性过滤（将不可行 pair 的分数设为 -inf） ──
        if feasibility_calibrator is not None:
            threshold = self.cfg.get("feasibility_threshold", 0.5)
            feas_mask = feasibility_calibrator.get_feasibility_mask(
                all_pairs, threshold
            )  # (P,)
            # 对 unseen pair 中不可行的屏蔽
            feas_mask_t = torch.tensor(feas_mask, dtype=torch.bool, device=all_scores.device)
            # seen pair 不受影响
            unseen_mask = torch.zeros(len(all_pairs), dtype=torch.bool)
            for i in unseen_idx_ow:
                unseen_mask[i] = True
            # 只对 unseen 应用可行性过滤
            infeasible_unseen = unseen_mask & ~feas_mask_t.cpu()
            all_scores[:, infeasible_unseen] = float("-inf")

        metrics = self._compute_metrics(
            all_scores = all_scores,
            gt_attr    = all_gt_attr,
            gt_obj     = all_gt_obj,
            pairs      = all_pairs,
            seen_idx   = seen_idx_ow,
            unseen_idx = unseen_idx_ow,
        )
        return metrics

    # ────────────────────────────
    # 分数收集 (提速版)
    # ────────────────────────────

    @torch.no_grad()
    def _collect_scores(
        self,
        dataloader,
        target_pairs: List[Tuple[str, str]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        对整个 dataloader 批次计算分数。
        """
        self.model.eval()

        all_scores  = []
        all_gt_attr = []
        all_gt_obj  = []

        # === 核心提速 1：在循环外一次性计算所有候选对的概念向量 ===
        tgt_attr_idx = torch.tensor(
            [self.dataset.attr2idx[a] for a, _ in target_pairs],
            dtype=torch.long, device=self.device
        )
        tgt_obj_idx  = torch.tensor(
            [self.dataset.obj2idx[o] for _, o in target_pairs],
            dtype=torch.long, device=self.device
        )
        # 调用昨天加了分块保护的计算方法
        concept_vecs = self.model.compute_concept_vectors(tgt_attr_idx, tgt_obj_idx)

        for batch in dataloader:
            images   = batch["image"].to(self.device)
            gt_attr  = batch["attr_idx"].tolist()
            gt_obj   = batch["obj_idx"].tolist()

            # 仅提取图像特征并直接做矩阵乘法，避开重复过文本模型的开销
            img_feats = self.model.clip.encode_image(images)
            scores = img_feats @ concept_vecs.T

            all_scores.append(scores.cpu())
            all_gt_attr.extend(gt_attr)
            all_gt_obj.extend(gt_obj)

        all_scores = torch.cat(all_scores, dim=0)  # (N, P)
        return all_scores, all_gt_attr, all_gt_obj

    # ────────────────────────────
    # 指标计算 (GPU 加速极速版)
    # ────────────────────────────

    def _compute_metrics(
        self,
        all_scores : torch.Tensor,      # (N, P)
        gt_attr    : List[int],
        gt_obj     : List[int],
        pairs      : List[Tuple[str, str]],
        seen_idx   : List[int],
        unseen_idx : List[int],
    ) -> Dict[str, float]:
        """
        通过扫描 bias，计算 S / U / HM / AUC。
        """
        attr2idx = self.dataset.attr2idx
        obj2idx  = self.dataset.obj2idx

        pair_attr = torch.tensor([attr2idx[a] for a, _ in pairs], dtype=torch.long)
        pair_obj  = torch.tensor([obj2idx[o]  for _, o in pairs], dtype=torch.long)

        gt_attr_t = torch.tensor(gt_attr, dtype=torch.long)  # (N,)
        gt_obj_t  = torch.tensor(gt_obj,  dtype=torch.long)  # (N,)

        gt_match = (
            (pair_attr.unsqueeze(0) == gt_attr_t.unsqueeze(1)) &
            (pair_obj.unsqueeze(0)  == gt_obj_t.unsqueeze(1))
        )  # (N, P) bool

        # 提前算好 seen / unseen 掩码，移出循环
        gt_pair_in_seen   = self._is_gt_in_subset(
            gt_attr_t, gt_obj_t, pairs, seen_idx, attr2idx, obj2idx
        )
        gt_pair_in_unseen = self._is_gt_in_subset(
            gt_attr_t, gt_obj_t, pairs, unseen_idx, attr2idx, obj2idx
        )

        seen_total = gt_pair_in_seen.float().sum().clamp(min=1)
        unseen_total = gt_pair_in_unseen.float().sum().clamp(min=1)

        # === 核心提速 2：将大矩阵全面转移到 GPU，利用张量并行取代 CPU 慢速计算 ===
        all_scores = all_scores.to(self.device)
        gt_match = gt_match.to(self.device)
        gt_pair_in_seen = gt_pair_in_seen.to(self.device)
        gt_pair_in_unseen = gt_pair_in_unseen.to(self.device)
        
        # 预先构建 Bias 偏置掩码，彻底抛弃巨耗内存的 .clone() 复制操作
        bias_mask = torch.zeros(len(pairs), device=self.device)
        bias_mask[unseen_idx] = 1.0

        seen_scores_list   = []   
        unseen_scores_list = []   

        for bias in self.biases:
            # 高效写法：原矩阵不动，利用广播机制加上偏置项并极速求出预测下标
            pred_idx = (all_scores + bias_mask * bias).argmax(dim=1)

            correct = gt_match[torch.arange(len(gt_attr), device=self.device), pred_idx]

            seen_acc   = (correct & gt_pair_in_seen).float().sum() / seen_total
            unseen_acc = (correct & gt_pair_in_unseen).float().sum() / unseen_total

            seen_scores_list.append(seen_acc.item())
            unseen_scores_list.append(unseen_acc.item())

        seen_arr   = np.array(seen_scores_list)
        unseen_arr = np.array(unseen_scores_list)
        hm_arr     = 2 * seen_arr * unseen_arr / (seen_arr + unseen_arr + 1e-8)

        auc = np.trapz(unseen_arr, seen_arr)
        if auc < 0:
            auc = -auc   

        best_hm_idx = hm_arr.argmax()

        return {
            "S"   : seen_arr.max() * 100,
            "U"   : unseen_arr.max() * 100,
            "HM"  : hm_arr.max() * 100,
            "AUC" : auc * 100,
            "best_bias_seen"  : seen_arr[best_hm_idx],
            "best_bias_unseen": unseen_arr[best_hm_idx],
        }

    @staticmethod
    def _is_gt_in_subset(
        gt_attr, gt_obj, pairs, subset_idx, attr2idx, obj2idx
    ) -> torch.Tensor:
        """判断每个样本的 gt pair 是否属于指定子集（seen/unseen）"""
        subset_pairs_attr = torch.tensor(
            [attr2idx[pairs[i][0]] for i in subset_idx], dtype=torch.long
        )
        subset_pairs_obj  = torch.tensor(
            [obj2idx[pairs[i][1]] for i in subset_idx], dtype=torch.long
        )
        # (N, 1) vs (1, |subset|)
        match = (
            (gt_attr.unsqueeze(1) == subset_pairs_attr.unsqueeze(0)) &
            (gt_obj.unsqueeze(1)  == subset_pairs_obj.unsqueeze(0))
        )  # (N, |subset|)
        return match.any(dim=1)   # (N,)