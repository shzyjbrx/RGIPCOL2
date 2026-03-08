"""
rgcn_encoder.py
---------------
多层 RGCN 堆叠，完成异构图上的节点特征传播。

主要职责
--------
1. 初始化节点嵌入：
     - attr / obj 节点：使用 CLIP 文本编码器对名称编码（冻结，只初始化一次）
     - comp 节点     ：初始化为对应 (attr_vec + obj_vec) / 2

2. 多层 RGCN 传播：
     x^(0) → RGCN_1 → ReLU → RGCN_2 → ... → x^(L)

3. 提取输出：
     返回更新后的 attr 节点嵌入和 obj 节点嵌入（供 PromptLearner 使用）
     comp 节点的嵌入作为中间媒介，训练中会反向传播梯度

注意：整个 RGCNEncoder 作为可训练模块插入主模型，
      CLIP 文本编码器用于初始化节点嵌入，初始化后不再调用（节省显存）。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgcn_layer import RGCNLayer


class RGCNEncoder(nn.Module):
    """
    异构图 RGCN 编码器。

    Parameters
    ----------
    clip_text_encoder : 冻结的 CLIP 文本编码函数（用于初始化节点嵌入）
    graph_data        : HeterogeneousGraphConstructor.build() 的返回字典
    hidden_dim        : RGCN 隐层维度（默认 768 与 CLIP 对齐）
    output_dim        : 输出维度（默认 768）
    num_layers        : RGCN 层数（默认 2）
    num_bases         : 基函数数量（-1 = 不用分解）
    dropout           : Dropout 概率
    """

    def __init__(
        self,
        clip_text_encoder,          # callable: List[str] → Tensor (N, D)
        graph_data    : dict,
        hidden_dim    : int   = 768,
        output_dim    : int   = 768,
        num_layers    : int   = 2,
        num_bases     : int   = -1,
        dropout       : float = 0.3,
    ):
        super().__init__()

        self.hidden_dim     = hidden_dim
        self.output_dim     = output_dim
        self.num_layers     = num_layers
        self.graph_data     = graph_data
        num_relations       = graph_data["num_relations"]   # 7

        # ── 1. 节点嵌入初始化（作为可学习参数，从 CLIP 嵌入热启动） ──
        node_init = self._init_node_embeddings(
            clip_text_encoder, graph_data, hidden_dim
        )
        # node_embeddings 是可学习参数，初始值来自 CLIP
        self.node_embeddings = nn.Parameter(node_init)   # (N, D)

        # ── 2. 输入维度投影（若 node_init 维度与 hidden_dim 不匹配） ──
        input_dim = node_init.size(1)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # ── 3. RGCN 层堆叠 ──
        layers = []
        for i in range(num_layers):
            in_d  = hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(
                RGCNLayer(
                    in_dim        = in_d,
                    out_dim       = out_d,
                    num_relations = num_relations,
                    num_bases     = num_bases,
                    aggr          = "mean",
                    bias          = True,
                    dropout       = dropout if i < num_layers - 1 else 0.0,
                )
            )
        self.rgcn_layers = nn.ModuleList(layers)

        # ── 4. 层归一化（每层后加，稳定训练） ──
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

        # ── 缓存图结构（不参与梯度） ──
        self.register_buffer("edge_index", graph_data["edge_index"])  # (2, E)
        self.register_buffer("edge_type",  graph_data["edge_type"])   # (E,)

        # 节点范围索引（用于提取 attr/obj 嵌入）
        self.num_attrs = graph_data["num_attrs"]
        self.num_objs  = graph_data["num_objs"]
        self.num_comps = graph_data["num_comps"]

    # ────────────────────────────
    # 节点嵌入初始化（仅调用一次）
    # ────────────────────────────

    @torch.no_grad()
    def _init_node_embeddings(
        self,
        clip_text_encoder,
        graph_data : dict,
        hidden_dim : int,
    ) -> torch.Tensor:
        """
        初始化所有节点嵌入（顺序：attr → obj → comp）。
        comp 节点 = (attr_emb + obj_emb) / 2
        """
        attrs       = graph_data["attrs"]
        objs        = graph_data["objs"]
        train_pairs = graph_data["train_pairs"]

        # ── attr 节点嵌入 ──
        attr_embs = clip_text_encoder(attrs)    # (A, D)
        # ── obj  节点嵌入 ──
        obj_embs  = clip_text_encoder(objs)     # (O, D)

        # ── comp 节点嵌入 = 平均 ──
        attr2nid_local = {a: i for i, a in enumerate(attrs)}
        obj2nid_local  = {o: i for i, o in enumerate(objs)}

        comp_embs_list = []
        for (attr, obj) in train_pairs:
            a_emb = attr_embs[attr2nid_local[attr]]
            o_emb = obj_embs[obj2nid_local[obj]]
            comp_embs_list.append((a_emb + o_emb) / 2.0)
        comp_embs = torch.stack(comp_embs_list, dim=0)  # (C, D)

        # ── 拼接 (A + O + C, D) ──
        node_init = torch.cat([attr_embs, obj_embs, comp_embs], dim=0)

        print(f"[RGCNEncoder] 节点嵌入初始化完成："
              f"attr={attr_embs.shape}, obj={obj_embs.shape}, "
              f"comp={comp_embs.shape}, total={node_init.shape}")

        return node_init.float()

    # ────────────────────────────
    # 前向传播
    # ────────────────────────────

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 RGCN 图传播，返回更新后的 attr 和 obj 节点嵌入。

        Returns
        -------
        attr_embs : (num_attrs, output_dim)  — 更新后的属性节点嵌入
        obj_embs  : (num_objs,  output_dim)  — 更新后的物体节点嵌入
        """
        # 起始节点特征：可学习 node_embeddings
        x = self.input_proj(self.node_embeddings)  # (N, hidden_dim)

        # 逐层传播
        for i, (layer, norm) in enumerate(zip(self.rgcn_layers, self.layer_norms)):
            x_new = layer(x, self.edge_index, self.edge_type)
            x_new = norm(x_new)
            # 残差连接（仅当维度匹配时）
            if x.size(-1) == x_new.size(-1):
                x = F.relu(x_new + x)
            else:
                x = F.relu(x_new)

        # 提取 attr / obj 节点嵌入
        attr_embs = x[: self.num_attrs]                              # (A, D)
        obj_embs  = x[self.num_attrs : self.num_attrs + self.num_objs]  # (O, D)

        return attr_embs, obj_embs

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.node_embeddings.size(0)}, "
            f"hidden={self.hidden_dim}, output={self.output_dim}, "
            f"layers={self.num_layers}"
        )