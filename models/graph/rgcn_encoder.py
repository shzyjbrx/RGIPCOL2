"""
rgcn_encoder.py
---------------
多层 RGCN 堆叠，完成异构图上的节点特征传播。带有双分支动态门控融合机制。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgcn_layer import RGCNLayer
from .llm_feature_loader import load_node_features


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        clip_text_encoder,          
        graph_data    : dict,
        hidden_dim    : int   = 768,
        output_dim    : int   = 768,
        num_layers    : int   = 2,
        num_bases     : int   = -1,
        dropout       : float = 0.3,
        llm_features_path : str = None,  # [修复 2] 添加缺失的路径参数
    ):
        super().__init__()

        self.hidden_dim     = hidden_dim
        self.output_dim     = output_dim
        self.num_layers     = num_layers
        self.graph_data     = graph_data
        num_relations       = graph_data["num_relations"]  

        # 节点范围索引提前，前向传播切分特征时需要用到
        self.num_attrs = graph_data["num_attrs"]
        self.num_objs  = graph_data["num_objs"]
        self.num_comps = graph_data["num_comps"]

        # ── 1. 节点双分支特征初始化 ──
        # [修复 1] 正确解包返回的元组，并注册为不可直接训练的 buffer
        h_clip, h_llm = self._init_node_embeddings(
            clip_text_encoder, graph_data, hidden_dim, llm_features_path
        )
        self.register_buffer("h_clip", h_clip)
        self.register_buffer("h_llm", h_llm)

        # ── 1.5 动态门控网络 (GateFusion) ──
        # [修复 3.1] 补充被遗漏的动态门控线性层
        self.gate_attr = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_obj  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_comp = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── 2. 输入维度投影 ──
        input_dim = h_clip.size(1)
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

        # ── 4. 层归一化 ──
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

        # ── 缓存图结构 ──
        self.register_buffer("edge_index", graph_data["edge_index"])  
        self.register_buffer("edge_type",  graph_data["edge_type"])   

    @torch.no_grad()
    def _init_node_embeddings(
        self,
        clip_text_encoder,
        graph_data : dict,
        hidden_dim : int,
        llm_features_path : str 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        attrs       = graph_data["attrs"]
        objs        = graph_data["objs"]
        train_pairs = graph_data["train_pairs"]
        
        comp_names = [f"{a} {o}" for a, o in train_pairs]

        # ── 1. CLIP 原始词汇特征 ──
        attr_clip = clip_text_encoder(attrs)
        obj_clip  = clip_text_encoder(objs)

        attr2nid_local = {a: i for i, a in enumerate(attrs)}
        obj2nid_local  = {o: i for i, o in enumerate(objs)}

        comp_clip_list = []
        for (attr, obj) in train_pairs:
            a_emb = attr_clip[attr2nid_local[attr]]
            o_emb = obj_clip[obj2nid_local[obj]]
            comp_clip_list.append((a_emb + o_emb) / 2.0)
            
        comp_clip = torch.stack(comp_clip_list, dim=0)
        h_clip = torch.cat([attr_clip, obj_clip, comp_clip], dim=0).float()

        # ── 2. LLM 离线描述特征 ──
        attr_llm = load_node_features(llm_features_path, attrs, clip_text_encoder)
        obj_llm  = load_node_features(llm_features_path, objs, clip_text_encoder)
        comp_llm = load_node_features(llm_features_path, comp_names, clip_text_encoder)

        h_llm = torch.cat([attr_llm, obj_llm, comp_llm], dim=0).float()

        print(f"[RGCNEncoder] 节点双分支特征初始化完成 (离线特征加载)：\n"
              f"  CLIP 名称特征: {h_clip.shape}\n"
              f"  LLM  描述特征: {h_llm.shape}")

        device = attr_clip.device
        return h_clip.to(device), h_llm.to(device)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # [修复 3.2] 在图传播之前，将 h_clip 与 h_llm 进行自适应门控融合
        clip_a, clip_o, clip_c = torch.split(self.h_clip, [self.num_attrs, self.num_objs, self.num_comps])
        llm_a, llm_o, llm_c    = torch.split(self.h_llm,  [self.num_attrs, self.num_objs, self.num_comps])

        # 分别计算三类节点的门控权重与特征融合
        g_a = torch.sigmoid(self.gate_attr(torch.cat([clip_a, llm_a], dim=-1)))
        h_a = g_a * clip_a + (1.0 - g_a) * llm_a

        g_o = torch.sigmoid(self.gate_obj(torch.cat([clip_o, llm_o], dim=-1)))
        h_o = g_o * clip_o + (1.0 - g_o) * llm_o

        g_c = torch.sigmoid(self.gate_comp(torch.cat([clip_c, llm_c], dim=-1)))
        h_c = g_c * clip_c + (1.0 - g_c) * llm_c

        # 拼接得到动态融合后的初始矩阵
        x = torch.cat([h_a, h_o, h_c], dim=0)
        x = self.input_proj(x)

        # 逐层传播
        for i, (layer, norm) in enumerate(zip(self.rgcn_layers, self.layer_norms)):
            x_new = layer(x, self.edge_index, self.edge_type)
            x_new = norm(x_new)
            if x.size(-1) == x_new.size(-1):
                x = F.relu(x_new + x)
            else:
                x = F.relu(x_new)

        # 提取 attr / obj 节点嵌入
        attr_embs = x[: self.num_attrs]                              
        obj_embs  = x[self.num_attrs : self.num_attrs + self.num_objs]  

        return attr_embs, obj_embs

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.h_clip.size(0)}, "
            f"hidden={self.hidden_dim}, output={self.output_dim}, "
            f"layers={self.num_layers}"
        )