"""
evaluator.py
------------
CZSL 标准评估器。

计算四个指标（闭/开世界均支持）：
  - S   : Best Seen Accuracy
  - U   : Best Unseen Accuracy
  - HM  : Best Harmonic Mean（扫描 bias 找最优）
  - AUC : Area Under Seen-Unseen Accuracy Curve

评估流程：
  1. 对所有图像，在候选 pair 集合上计算相似度分数
  2. 应用可行性掩码（若提供）—— 双阶段过滤在此生效
  3. 对 unseen pair 批量添加 bias（从 -1 到 +1 扫描）
  4. 计算每个 bias 下的 seen_acc / unseen_acc
  5. 计算 S / U / HM / AUC

可行性校准器接口规范：
  任何校准器（TwoStageFeasibilityCalibrator / FeasibilityCalibrator）
  均须实现 get_feasibility_mask(pairs) -> np.ndarray[bool]
  thresholds 在校准器构造时传入，此处无需再传。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


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
        seen_idx   = [i for i, p in enumerate(self.closed_pairs)
                      if p in self.seen_set]
        unseen_idx = [i for i, p in enumerate(self.closed_pairs)
                      if p in self.unseen_set]
        return self._evaluate_efficient(
            dataloader, self.closed_pairs, seen_idx, unseen_idx
        )

    # ────────────────────────────
    # 开放世界评估
    # ────────────────────────────

    def evaluate_open_world(
        self,
        dataloader,
        feasibility_calibrator=None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        feasibility_calibrator : TwoStageFeasibilityCalibrator 或 FeasibilityCalibrator 实例
                                 调用其 get_feasibility_mask(pairs) 获取布尔掩码。
                                 None 表示不进行可行性过滤（全空间评估）。
        """
        all_pairs  = self.dataset.all_pairs
        seen_idx   = [i for i, p in enumerate(all_pairs) if p in self.seen_set]
        unseen_idx = [i for i, p in enumerate(all_pairs) if p in self.unseen_set]

        feas_mask_t = None
        if feasibility_calibrator is not None:
            print(f"[Evaluator] 应用可行性掩码，候选 pair 总数：{len(all_pairs)}")
            # ── 统一接口：不再传 threshold，由校准器内部维护 ──
            feas_mask_np = feasibility_calibrator.get_feasibility_mask(all_pairs)
            feas_mask_t  = torch.tensor(
                feas_mask_np, dtype=torch.bool, device=self.device
            )
            n_keep = int(feas_mask_np.sum())
            n_seen_kept = int(feas_mask_np[seen_idx].sum())
            n_unseen_kept = int(feas_mask_np[unseen_idx].sum())
            print(
                f"[Evaluator] 可行性过滤后保留 {n_keep}/{len(all_pairs)} pair "
                f"（seen={n_seen_kept}/{len(seen_idx)}, "
                f"unseen={n_unseen_kept}/{len(unseen_idx)}）"
            )
        else:
            print(f"[Evaluator] 未提供可行性校准器，全空间开放世界评估")

        return self._evaluate_efficient(
            dataloader, all_pairs, seen_idx, unseen_idx, feas_mask_t
        )

    # ────────────────────────────
    # 核心评估引擎（OOM 免疫版）
    # ────────────────────────────

    @torch.no_grad()
    def _evaluate_efficient(
        self,
        dataloader,
        target_pairs : List[Tuple[str, str]],
        seen_idx     : List[int],
        unseen_idx   : List[int],
        feas_mask    : Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:

        self.model.eval()

        # ── 计算候选对概念向量 ──
        tgt_attr_idx = torch.tensor(
            [self.dataset.attr2idx[a] for a, _ in target_pairs],
            dtype=torch.long, device=self.device
        )
        tgt_obj_idx = torch.tensor(
            [self.dataset.obj2idx[o] for _, o in target_pairs],
            dtype=torch.long, device=self.device
        )
        concept_vecs = self.model.compute_concept_vectors(tgt_attr_idx, tgt_obj_idx)

        seen_idx_t   = torch.tensor(seen_idx,   dtype=torch.long, device=self.device)
        unseen_idx_t = torch.tensor(unseen_idx, dtype=torch.long, device=self.device)

        # ── 逐 batch 前向，只保留极小量统计量 ──
        list_max_seen, list_max_unseen           = [], []
        list_gt_is_max_seen, list_gt_is_max_unseen = [], []
        list_gt_is_seen, list_gt_is_unseen       = [], []

        for batch in dataloader:
            images  = batch["image"].to(self.device)
            gt_attr = batch["attr_idx"].to(self.device)
            gt_obj  = batch["obj_idx"].to(self.device)

            img_feats = self.model.clip.encode_image(images)
            scores    = img_feats @ concept_vecs.T   # (B, P)

            # ── 应用可行性掩码：不可行 pair 分数设为 -inf ──
            if feas_mask is not None:
                invalid_mask = ~feas_mask
                # seen pair 始终可见（即使可行性分数低）
                invalid_mask[seen_idx_t] = False
                scores = scores.masked_fill(
                    invalid_mask.unsqueeze(0), float("-inf")
                )

            # ── 找 GT 匹配矩阵 ──
            pair_attr      = tgt_attr_idx.unsqueeze(0) == gt_attr.unsqueeze(1)
            pair_obj       = tgt_obj_idx.unsqueeze(0)  == gt_obj.unsqueeze(1)
            gt_match       = pair_attr & pair_obj   # (B, P)

            gt_scores         = torch.where(
                gt_match, scores,
                torch.tensor(float("-inf"), device=self.device)
            ).max(dim=1)[0]

            max_seen_scores   = scores[:, seen_idx_t].max(dim=1)[0]
            max_unseen_scores = scores[:, unseen_idx_t].max(dim=1)[0]

            gt_is_max_seen    = (gt_scores == max_seen_scores)
            gt_is_max_unseen  = (gt_scores == max_unseen_scores)
            gt_is_seen        = gt_match[:, seen_idx_t].any(dim=1)
            gt_is_unseen      = gt_match[:, unseen_idx_t].any(dim=1)

            # 转 CPU，释放大矩阵显存
            list_max_seen.append(max_seen_scores.cpu())
            list_max_unseen.append(max_unseen_scores.cpu())
            list_gt_is_max_seen.append(gt_is_max_seen.cpu())
            list_gt_is_max_unseen.append(gt_is_max_unseen.cpu())
            list_gt_is_seen.append(gt_is_seen.cpu())
            list_gt_is_unseen.append(gt_is_unseen.cpu())

        # ── 合并 ──
        max_seen          = torch.cat(list_max_seen).to(self.device)
        max_unseen        = torch.cat(list_max_unseen).to(self.device)
        gt_is_max_seen    = torch.cat(list_gt_is_max_seen).to(self.device)
        gt_is_max_unseen  = torch.cat(list_gt_is_max_unseen).to(self.device)
        gt_is_seen        = torch.cat(list_gt_is_seen).to(self.device)
        gt_is_unseen      = torch.cat(list_gt_is_unseen).to(self.device)

        num_seen_gt   = gt_is_seen.sum().clamp(min=1).item()
        num_unseen_gt = gt_is_unseen.sum().clamp(min=1).item()

        # ── Bias 扫描 ──
        seen_scores_list, unseen_scores_list = [], []

        for bias in self.biases:
            correct_seen = (
                gt_is_seen &
                gt_is_max_seen &
                (max_seen >= max_unseen + bias)
            )
            correct_unseen = (
                gt_is_unseen &
                gt_is_max_unseen &
                (max_unseen + bias > max_seen)
            )
            seen_scores_list.append(
                (correct_seen.float().sum() / num_seen_gt).item()
            )
            unseen_scores_list.append(
                (correct_unseen.float().sum() / num_unseen_gt).item()
            )

        seen_arr   = np.array(seen_scores_list)
        unseen_arr = np.array(unseen_scores_list)
        hm_arr     = (2 * seen_arr * unseen_arr
                      / (seen_arr + unseen_arr + 1e-8))

        auc = float(np.trapz(unseen_arr, seen_arr))
        if auc < 0:
            auc = -auc

        best_hm_idx = hm_arr.argmax()

        return {
            "S"               : float(seen_arr.max())   * 100,
            "U"               : float(unseen_arr.max()) * 100,
            "HM"              : float(hm_arr.max())     * 100,
            "AUC"             : auc * 100,
            "best_bias_seen"  : float(seen_arr[best_hm_idx]),
            "best_bias_unseen": float(unseen_arr[best_hm_idx]),
        }