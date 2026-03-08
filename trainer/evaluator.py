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

"""
evaluator.py
------------
CZSL 极速/极低内存评估器。
利用群组最大值（Group-Max）降维算法，将内存占用缩减 99.9%，彻底解决 OOM。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from trainer.feasibility import FeasibilityCalibrator


class Evaluator:
    def __init__(self, model, dataset, cfg: dict, device: torch.device):
        self.model   = model
        self.dataset = dataset
        self.cfg     = cfg
        self.device  = device

        self.closed_pairs = dataset.closed_pairs        
        self.seen_set     = set(dataset.seen_pairs)
        self.unseen_set   = set(dataset.unseen_pairs)

        bias_step = 0.001 
        self.biases = np.arange(-1.0, 1.0 + bias_step, bias_step)

    # ────────────────────────────
    # 闭合世界评估
    # ────────────────────────────
    def evaluate_closed_world(self, dataloader) -> Dict[str, float]:
        seen_idx = [i for i, p in enumerate(self.closed_pairs) if p in self.seen_set]
        unseen_idx = [i for i, p in enumerate(self.closed_pairs) if p in self.unseen_set]

        return self._evaluate_efficient(dataloader, self.closed_pairs, seen_idx, unseen_idx)

    # ────────────────────────────
    # 开放世界评估
    # ────────────────────────────
    def evaluate_open_world(
        self,
        dataloader,
        feasibility_calibrator: Optional[FeasibilityCalibrator] = None,
    ) -> Dict[str, float]:
        all_pairs = self.dataset.all_pairs

        seen_idx   = [i for i, p in enumerate(all_pairs) if p in self.seen_set]
        unseen_idx = [i for i, p in enumerate(all_pairs) if p in self.unseen_set]

        feas_mask_t = None
        if feasibility_calibrator is not None:
            threshold = self.cfg.get("feasibility_threshold", 0.5)
            feas_mask_np = feasibility_calibrator.get_feasibility_mask(all_pairs, threshold)
            feas_mask_t = torch.tensor(feas_mask_np, dtype=torch.bool, device=self.device)

        return self._evaluate_efficient(dataloader, all_pairs, seen_idx, unseen_idx, feas_mask_t)

    # ────────────────────────────
    # 核心引擎：OOM 免疫版
    # ────────────────────────────
    @torch.no_grad()
    def _evaluate_efficient(
        self,
        dataloader,
        target_pairs: List[Tuple[str, str]],
        seen_idx: List[int],
        unseen_idx: List[int],
        feas_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        
        self.model.eval()

        tgt_attr_idx = torch.tensor([self.dataset.attr2idx[a] for a, _ in target_pairs], dtype=torch.long, device=self.device)
        tgt_obj_idx  = torch.tensor([self.dataset.obj2idx[o] for _, o in target_pairs], dtype=torch.long, device=self.device)
        concept_vecs = self.model.compute_concept_vectors(tgt_attr_idx, tgt_obj_idx)

        seen_idx_t = torch.tensor(seen_idx, dtype=torch.long, device=self.device)
        unseen_idx_t = torch.tensor(unseen_idx, dtype=torch.long, device=self.device)

        list_max_seen, list_max_unseen = [], []
        list_gt_is_max_seen, list_gt_is_max_unseen = [], []
        list_gt_is_seen, list_gt_is_unseen = [], []

        for batch in dataloader:
            images  = batch["image"].to(self.device)
            gt_attr = batch["attr_idx"].to(self.device)
            gt_obj  = batch["obj_idx"].to(self.device)

            img_feats = self.model.clip.encode_image(images)
            scores = img_feats @ concept_vecs.T  # (B, P)

            # --- 1. 应用可行性屏蔽 ---
            if feas_mask is not None:
                invalid_mask = (~feas_mask)
                invalid_mask[seen_idx_t] = False  # 永远不要屏蔽真实的 seen 数据
                scores = scores.masked_fill(invalid_mask.unsqueeze(0), float("-inf"))

            # --- 2. 获取 GT 相关掩码 ---
            pair_attr = tgt_attr_idx.unsqueeze(0) == gt_attr.unsqueeze(1)
            pair_obj  = tgt_obj_idx.unsqueeze(0) == gt_obj.unsqueeze(1)
            gt_match_matrix = pair_attr & pair_obj  # (B, P)

            # 获取当前样本 GT 的分数
            gt_scores = torch.where(gt_match_matrix, scores, torch.tensor(float('-inf'), device=self.device)).max(dim=1)[0]

            # 获取 seen 和 unseen 各自的最高分
            max_seen_scores = scores[:, seen_idx_t].max(dim=1)[0]
            max_unseen_scores = scores[:, unseen_idx_t].max(dim=1)[0]

            # 记录真实标签是否拿到了所属类别的最高分
            gt_is_max_seen = (gt_scores == max_seen_scores)
            gt_is_max_unseen = (gt_scores == max_unseen_scores)

            # 记录真实标签是属于 seen 还是 unseen
            gt_is_seen = gt_match_matrix[:, seen_idx_t].any(dim=1)
            gt_is_unseen = gt_match_matrix[:, unseen_idx_t].any(dim=1)

            # 将极小体积的 1D 张量转移回 CPU，当场释放巨大 scores 矩阵的显存和内存
            list_max_seen.append(max_seen_scores.cpu())
            list_max_unseen.append(max_unseen_scores.cpu())
            list_gt_is_max_seen.append(gt_is_max_seen.cpu())
            list_gt_is_max_unseen.append(gt_is_max_unseen.cpu())
            list_gt_is_seen.append(gt_is_seen.cpu())
            list_gt_is_unseen.append(gt_is_unseen.cpu())

        # 拼接小张量
        max_seen = torch.cat(list_max_seen).to(self.device)
        max_unseen = torch.cat(list_max_unseen).to(self.device)
        gt_is_max_seen = torch.cat(list_gt_is_max_seen).to(self.device)
        gt_is_max_unseen = torch.cat(list_gt_is_max_unseen).to(self.device)
        gt_is_seen = torch.cat(list_gt_is_seen).to(self.device)
        gt_is_unseen = torch.cat(list_gt_is_unseen).to(self.device)

        num_seen_gt = gt_is_seen.sum().clamp(min=1).item()
        num_unseen_gt = gt_is_unseen.sum().clamp(min=1).item()

        seen_scores_list = []
        unseen_scores_list = []

        # --- 3. 极速 Bias 扫描 ---
        for bias in self.biases:
            # 如果 GT 是 seen，且它在 seen 里是老大，并且它的分数干掉了加上 bias 的 unseen 老大，则预测正确
            correct_seen = gt_is_seen & gt_is_max_seen & (max_seen >= max_unseen + bias)
            # 如果 GT 是 unseen，且它在 unseen 里是老大，并且它加上 bias 后干掉了 seen 老大，则预测正确
            correct_unseen = gt_is_unseen & gt_is_max_unseen & (max_unseen + bias > max_seen)

            seen_acc = correct_seen.float().sum() / num_seen_gt
            unseen_acc = correct_unseen.float().sum() / num_unseen_gt

            seen_scores_list.append(seen_acc.item())
            unseen_scores_list.append(unseen_acc.item())

        seen_arr = np.array(seen_scores_list)
        unseen_arr = np.array(unseen_scores_list)
        hm_arr = 2 * seen_arr * unseen_arr / (seen_arr + unseen_arr + 1e-8)

        auc = np.trapz(unseen_arr, seen_arr)
        if auc < 0: auc = -auc

        best_hm_idx = hm_arr.argmax()

        return {
            "S": seen_arr.max() * 100,
            "U": unseen_arr.max() * 100,
            "HM": hm_arr.max() * 100,
            "AUC": auc * 100,
            "best_bias_seen": seen_arr[best_hm_idx],
            "best_bias_unseen": unseen_arr[best_hm_idx],
        }