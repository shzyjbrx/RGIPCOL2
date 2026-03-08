"""
rgcn_layer.py
-------------
单层 Relational Graph Convolutional Network (RGCN)。

标准 RGCN 公式（Schlichtkrull et al., 2018）：

    h_i^(l+1) = σ(
        Σ_r  Σ_{j ∈ N_r(i)}  (1 / c_{i,r}) * W_r^(l) * h_j^(l)
        + W_0^(l) * h_i^(l)
    )

其中：
  - r     : 关系类型（边类型）
  - N_r(i): 节点 i 在关系 r 下的邻居集合
  - c_{i,r}: 归一化常数（= |N_r(i)|，即该关系下邻居数）
  - W_r   : 每种关系独立的可学习权重矩阵（in_dim → out_dim）
  - W_0   : 自连接权重矩阵（处理 self-loop，若 self-loop 已作为一种
             关系类型放入 edge_type，则 W_0 可复用 W_r[SELF_LOOP]）

基函数分解（Basis Decomposition，防止参数过多导致过拟合）：
    W_r = Σ_b  a_{rb} * V_b
  其中 V_b 是共享基矩阵，a_{rb} 是可学习系数。
  当 num_bases == -1 时，不使用分解，直接学习每个 W_r。

实现说明
--------
本实现采用"稀疏聚合"策略：
  1. 对每种关系 r，收集该关系下的所有 (src, dst) 对
  2. 用 W_r 变换 src 节点特征
  3. 按 dst 聚合（sum / mean）
  4. 汇总所有关系的结果
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    """
    单层 RGCN。

    Parameters
    ----------
    in_dim       : 输入特征维度
    out_dim      : 输出特征维度
    num_relations: 关系类型数（= NUM_RELATIONS = 7）
    num_bases    : 基函数数量；-1 表示不使用基函数分解
    aggr         : 聚合方式，"sum" | "mean"（默认 "mean"）
    bias         : 是否添加偏置项
    dropout      : Dropout 概率（作用于聚合后、激活前）
    """

    def __init__(
        self,
        in_dim       : int,
        out_dim      : int,
        num_relations: int,
        num_bases    : int  = -1,
        aggr         : str  = "mean",
        bias         : bool = True,
        dropout      : float = 0.0,
    ):
        super().__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_relations = num_relations
        self.num_bases     = num_bases
        self.aggr          = aggr

        # ── 权重矩阵 ──
        use_basis = (num_bases > 0) and (num_bases < num_relations)

        if use_basis:
            # 基函数分解：V_b (num_bases, in_dim, out_dim)
            #             a_rb (num_relations, num_bases)
            self.basis       = nn.Parameter(
                torch.empty(num_bases, in_dim, out_dim)
            )
            self.coeff       = nn.Parameter(
                torch.empty(num_relations, num_bases)
            )
            self._use_basis  = True
        else:
            # 每种关系独立的 W_r：(num_relations, in_dim, out_dim)
            self.weight      = nn.Parameter(
                torch.empty(num_relations, in_dim, out_dim)
            )
            self._use_basis  = False

        # 偏置（一个全局偏置，作用于所有关系聚合后的输出）
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier 均匀初始化权重"""
        if self._use_basis:
            nn.init.xavier_uniform_(self.basis)
            nn.init.xavier_uniform_(self.coeff)
        else:
            nn.init.xavier_uniform_(self.weight.view(self.num_relations, -1)
                                    .unsqueeze(0))
            # xavier_uniform_ 要求 2D，绕过方式：
            for i in range(self.num_relations):
                nn.init.xavier_uniform_(self.weight[i])

    # ────────────────────────────
    # 前向传播
    # ────────────────────────────

    # ────────────────────────────
    # 前向传播 (修复 OOM 版)
    # ────────────────────────────

    def forward(
        self,
        x         : torch.Tensor,   # (N, in_dim) — 所有节点特征
        edge_index: torch.Tensor,   # (2, E)       — [src; dst]
        edge_type : torch.Tensor,   # (E,)          — 每条边的关系类型
    ) -> torch.Tensor:
        """
        Returns
        -------
        out : (N, out_dim)
        """
        N   = x.size(0)
        device = x.device

        # ── 计算各关系的权重矩阵 W_r ──
        if self._use_basis:
            # W_r = Σ_b a_{rb} * V_b
            # coeff: (R, B), basis: (B, in, out) → weight: (R, in, out)
            weight = torch.einsum("rb,bio->rio", self.coeff, self.basis)
        else:
            weight = self.weight  # (R, in_dim, out_dim)

        src_nodes = edge_index[0]  # (E,)
        dst_nodes = edge_index[1]  # (E,)

        # 初始化输出特征矩阵
        out = torch.zeros(N, self.out_dim, device=device, dtype=x.dtype)

        # ── 核心修复：按关系类型进行循环计算，极大节省显存 ──
        for r in range(self.num_relations):
            # 筛选出属于当前关系 r 的边
            mask = (edge_type == r)
            if not mask.any():
                continue

            # 拿到当前关系下的起点和终点
            src_r = src_nodes[mask]  # (E_r,)
            dst_r = dst_nodes[mask]  # (E_r,)

            # 拿到当前关系下的源节点特征: (E_r, in_dim)
            h_src_r = x[src_r]
            
            # 拿到当前关系专属的变换矩阵: (in_dim, out_dim)
            W_r = weight[r]

            # 矩阵乘法: (E_r, in_dim) @ (in_dim, out_dim) -> (E_r, out_dim)
            # 这里只需分配 (E_r, 768) 的显存，极小！
            msg_r = torch.matmul(h_src_r, W_r)
            msg_r = self.dropout(msg_r)

            # 按 dst 聚合 (Scatter Add) 到全局 out 张量中
            out.scatter_add_(0, dst_r.unsqueeze(1).expand_as(msg_r), msg_r)

        # ── 按 dst 归一化 (Mean 聚合) ──
        if self.aggr == "mean":
            # 计算每个节点的入度（各关系合计）
            deg = torch.zeros(N, device=device, dtype=x.dtype)
            deg.scatter_add_(0, dst_nodes, torch.ones(dst_nodes.size(0),
                                                       device=device,
                                                       dtype=x.dtype))
            # 避免除零（孤立节点）
            deg = deg.clamp(min=1.0)
            out = out / deg.unsqueeze(1)

        # 加偏置
        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_dim}, out={self.out_dim}, "
            f"relations={self.num_relations}, "
            f"bases={'None' if not self._use_basis else self.num_bases}, "
            f"aggr={self.aggr}"
        )