"""
rgipcol.py
----------
RGIPCOL 主模型：将 CLIPAdapter / RGCNEncoder / PromptLearner 整合为完整系统。

前向流程：
  1. RGCN 图传播     → 更新后的 attr/obj 节点嵌入
  2. PromptLearner   → 为所有候选对构建软提示 → 组合概念向量 {c_i}
  3. CLIP 视觉编码器 → 图像特征向量 x
  4. Cosine 相似度   → p(c_i | x)
  5. 正则化交叉熵损失

推理流程：
  - 闭合世界：对 seen ∪ unseen 对计算 cosine，argmax 取最优
  - 开放世界：全部 attr×obj 对，经可行性阈值过滤后 argmax
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_adapter import CLIPAdapter
from .graph.graph_constructor import HeterogeneousGraphConstructor
from .graph.rgcn_encoder import RGCNEncoder
from .prompt_learner import PromptLearner


class RGIPCOL(nn.Module):
    """
    RGIPCOL 主模型。

    Parameters
    ----------
    cfg     : 配置字典（来自 yaml）
    dataset : CompositionDataset 实例（提供 attrs/objs/train_pairs 等信息）
    device  : 运行设备
    """

    def __init__(self, cfg: dict, dataset, device: torch.device):
        super().__init__()
        

        self.cfg    = cfg
        self.device = device

        # ── 1. CLIP 适配器（冻结） ──
        self.clip = CLIPAdapter(
            model_name = cfg["clip_model"],
            model_path = cfg.get("clip_model_path"),
            device     = device,
        )
        feat_dim = self.clip.feature_dim

        # ── 2. 构建异构图 ──
        cache_path = os.path.join(
            cfg.get("graph_cache_dir", "./graph_data"),
            f"{cfg['dataset']}_hetero_graph.pt"
        )
        constructor = HeterogeneousGraphConstructor(
            attrs       = dataset.attrs,
            objs        = dataset.objs,
            train_pairs = dataset.train_pairs,
            cache_path  = cache_path,
        )
        graph_data = constructor.build()

        # 将图数据移动到目标设备
        graph_data["edge_index"] = graph_data["edge_index"].to(device)
        graph_data["edge_type"]  = graph_data["edge_type"].to(device)

        # ── 3. RGCN 编码器（可学习） ──
        self.rgcn = RGCNEncoder(
            clip_text_encoder = self.clip.encode_words,
            graph_data        = graph_data,
            hidden_dim        = cfg.get("rgcn_hidden_dim", feat_dim),
            output_dim        = cfg.get("rgcn_output_dim", feat_dim),
            num_layers        = cfg.get("rgcn_num_layers", 2),
            num_bases         = cfg.get("rgcn_num_bases", -1),
            dropout           = cfg.get("rgcn_dropout", 0.3),
        ).to(device)

        # ── 4. 软提示学习器（可学习） ──
        self.prompt_learner = PromptLearner(
            clip_adapter  = self.clip,
            prefix_length = cfg.get("prefix_length", 3),
            feature_dim   = feat_dim,
        ).to(device)

        # ── 5. 缓存数据集信息 ──
        self.attrs       = dataset.attrs
        self.objs        = dataset.objs
        self.train_pairs = dataset.train_pairs   # seen pairs（训练时用）

        self.attr2idx      = dataset.attr2idx
        self.obj2idx       = dataset.obj2idx
        self.train_pair2idx = dataset.train_pair2idx

        # ── 6. 预构建索引张量（推理时用） ──
        # 为每个 seen pair 预存 attr_idx / obj_idx（用于批量从 RGCN 输出切片）
        train_attr_indices = torch.tensor(
            [self.attr2idx[a] for a, _ in self.train_pairs], dtype=torch.long
        )
        train_obj_indices  = torch.tensor(
            [self.obj2idx[o]  for _, o in self.train_pairs], dtype=torch.long
        )
        self.register_buffer("train_attr_indices", train_attr_indices)
        self.register_buffer("train_obj_indices",  train_obj_indices)

        # ── 7. 打印参数统计 ──
        self._print_param_stats()

    # ────────────────────────────
    # 核心：计算所有训练对的概念向量
    # ────────────────────────────

    def compute_concept_vectors(
        self,
        pairs_attr_idx : Optional[torch.Tensor] = None,
        pairs_obj_idx  : Optional[torch.Tensor] = None,
        chunk_size     : int = 1024  # 新增：分块大小，显存不够可以改成 512
    ) -> torch.Tensor:
        """
        通过 RGCN + PromptLearner 计算组合概念向量 (支持分块防 OOM)。
        """
        # RGCN 传播（一次图前向传播计算全局节点更新，开销很小）
        attr_embs_all, obj_embs_all = self.rgcn()

        if pairs_attr_idx is None:
            pairs_attr_idx = self.train_attr_indices   # (num_train_pairs,)
        if pairs_obj_idx is None:
            pairs_obj_idx  = self.train_obj_indices    # (num_train_pairs,)

        # 按 pair 索引切片
        attr_embs = attr_embs_all[pairs_attr_idx]  # (P, D)
        obj_embs  = obj_embs_all[pairs_obj_idx]    # (P, D)

        num_pairs = attr_embs.size(0)

        # === 核心修改：分块防止 OOM ===
        if num_pairs <= chunk_size or self.training:
            # 训练时通常组合数量较少（如只有 train_pairs），直接一次性计算以保持计算图完整
            concept_vecs = self.prompt_learner(attr_embs, obj_embs)
        else:
            # 推理时（开放世界有数万个组合），进行切块（Chunking）处理
            concept_vecs_list = []
            for i in range(0, num_pairs, chunk_size):
                end_i = min(i + chunk_size, num_pairs)
                chunk_a = attr_embs[i:end_i]
                chunk_o = obj_embs[i:end_i]
                
                # 逐块送入 CLIP Text Encoder
                chunk_vecs = self.prompt_learner(chunk_a, chunk_o)
                concept_vecs_list.append(chunk_vecs)
                
            # 将所有分块结果重新拼接回 (P, D)
            concept_vecs = torch.cat(concept_vecs_list, dim=0)

        return concept_vecs

    # ────────────────────────────
    # 前向传播（训练阶段）
    # ────────────────────────────

    def forward(
        self,
        images     : torch.Tensor,   # (B, 3, H, W)
        pair_labels: torch.Tensor,   # (B,)  训练对索引（在 train_pairs 中的位置）
    ) -> torch.Tensor:
        """
        计算训练损失。

        Parameters
        ----------
        images      : (B, 3, H, W)  预处理后的图像
        pair_labels : (B,)          每张图像对应的 train_pair 索引

        Returns
        -------
        loss : scalar Tensor
        """
        # ── 图像特征（冻结 CLIP 视觉编码器） ──
        img_feats = self.clip.encode_image(images)    # (B, D)，已 L2 归一化

        # ── 所有训练对的组合概念向量 ──
        concept_vecs = self.compute_concept_vectors()  # (K, D)  K=num_train_pairs

        # ── Cosine 相似度（使用 CLIP 的温度缩放） ──
        logit_scale = self.clip.logit_scale             # scalar
        # (B, D) × (D, K) → (B, K)
        logits = logit_scale * img_feats @ concept_vecs.T

        # ── 正则化交叉熵损失（公式 7） ──
        ce_loss = F.cross_entropy(logits, pair_labels)

        lambda1 = self.cfg.get("lambda_prefix", 1e-5)
        lambda2 = self.cfg.get("lambda_rgcn",   1e-5)

        reg_prefix = lambda1 * self.prompt_learner.prefix_vectors.norm(p=2)
        reg_rgcn   = lambda2 * sum(
            p.norm(p=2)
            for p in self.rgcn.rgcn_layers.parameters()
        )

        loss = ce_loss + reg_prefix + reg_rgcn
        return loss

    # ────────────────────────────
    # 推理阶段
    # ────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images         : torch.Tensor,          # (B, 3, H, W)
        target_pairs   : List[Tuple[str, str]], # 候选组合对列表（闭/开世界）
        feasibility_scores: Optional[Dict[Tuple, float]] = None,  # 开放世界可行性分数
        feasibility_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对图像批次在候选 pair 集合上进行预测。

        Parameters
        ----------
        images               : (B, 3, H, W)
        target_pairs         : P 个候选 (attr, obj) 对
        feasibility_scores   : {(attr, obj): float}，None 表示闭合世界
        feasibility_threshold: 可行性过滤阈值

        Returns
        -------
        scores   : (B, P)  每个 pair 的相似度分数
        pred_idx : (B,)    预测的 pair 索引（在 target_pairs 中）
        """
        # ── 构建目标对的索引 ──
        tgt_attr_idx = torch.tensor(
            [self.attr2idx[a] for a, _ in target_pairs],
            dtype=torch.long, device=self.device
        )
        tgt_obj_idx  = torch.tensor(
            [self.obj2idx[o] for _, o in target_pairs],
            dtype=torch.long, device=self.device
        )

        # ── 计算概念向量 ──
        concept_vecs = self.compute_concept_vectors(tgt_attr_idx, tgt_obj_idx)
        # (P, D)

        # ── 图像特征 ──
        img_feats = self.clip.encode_image(images)   # (B, D)

        # ── 相似度分数 ──
        logit_scale = self.clip.logit_scale
        # scores = logit_scale * img_feats @ concept_vecs.T  # (B, P)
        scores = img_feats @ concept_vecs.T

        # ── 开放世界：应用可行性过滤 ──
        if feasibility_scores is not None:
            feasibility_mask = torch.tensor(
                [feasibility_scores.get(p, 0.0) >= feasibility_threshold
                 for p in target_pairs],
                dtype=torch.bool, device=self.device
            )  # (P,)
            # 将不可行 pair 的分数设为 -inf
            scores = scores.masked_fill(~feasibility_mask.unsqueeze(0), float("-inf"))

        pred_idx = scores.argmax(dim=-1)   # (B,)
        return scores, pred_idx

    # ────────────────────────────
    # 参数统计
    # ────────────────────────────

    def _print_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen       = total_params - trainable

        print("=" * 55)
        print("[RGIPCOL] 参数统计")
        print(f"  总参数量   : {total_params:,}")
        print(f"  可训练参数 : {trainable:,}")
        print(f"  冻结参数   : {frozen:,}")
        print(f"  前缀向量   : {self.prompt_learner.prefix_vectors.numel():,}")
        rgcn_params = sum(p.numel() for p in self.rgcn.parameters() if p.requires_grad)
        print(f"  RGCN 参数  : {rgcn_params:,}")
        print("=" * 55)