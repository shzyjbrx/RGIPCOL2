"""
graph_constructor.py
--------------------
构建 CZSL 异构有向图。

节点类型（3种）：
  - attr 节点  : 索引 [0, num_attrs)
  - obj  节点  : 索引 [num_attrs, num_attrs + num_objs)
  - comp 节点  : 索引 [num_attrs + num_objs, num_attrs + num_objs + num_train_pairs)

有向边类型（7种，对应 EdgeType 枚举）：
  0  SELF_LOOP   : v → v            （所有节点）
  1  ATTR2COMP   : attr → comp      （属性塑造组合）
  2  COMP2ATTR   : comp → attr      （组合反馈属性上下文）
  3  OBJ2COMP    : obj  → comp      （物体承载组合）
  4  COMP2OBJ    : comp → obj       （组合反馈物体状态）
  5  ATTR2OBJ    : attr → obj       （属性作用于物体）
  6  OBJ2ATTR    : obj  → attr      （物体约束属性表现形式）

GIPCOL 原版仅有 3 类无向边（等价于本设计的 1,2,3,4 四类 + self-loop，但无向意味着
方向信息丢失）。本设计将每类无向边拆成两条有向边，并额外补充 ATTR2OBJ / OBJ2ATTR，
总计 7 类，由 RGCN 对每类使用独立权重矩阵建模。
"""

import os
import pickle
from enum import IntEnum
from typing import List, Tuple

import torch


# ──────────────────────────────────────────────────────────────
# 边类型枚举
# ──────────────────────────────────────────────────────────────

class EdgeType(IntEnum):
    SELF_LOOP  = 0   # v → v
    ATTR2COMP  = 1   # attr → comp
    COMP2ATTR  = 2   # comp → attr
    OBJ2COMP   = 3   # obj  → comp
    COMP2OBJ   = 4   # comp → obj
    ATTR2OBJ   = 5   # attr → obj
    OBJ2ATTR   = 6   # obj  → attr

NUM_RELATIONS = len(EdgeType)   # = 7


# ──────────────────────────────────────────────────────────────
# 主构建器
# ──────────────────────────────────────────────────────────────

class HeterogeneousGraphConstructor:
    """
    给定数据集的 attrs / objs / train_pairs，构建异构有向图。

    主要输出（均为 torch.Tensor）：
      - node_ids_attr  : attr 节点的全局索引  shape (num_attrs,)
      - node_ids_obj   : obj  节点的全局索引  shape (num_objs,)
      - node_ids_comp  : comp 节点的全局索引  shape (num_train_pairs,)
      - edge_index     : shape (2, num_edges) —— [src_nodes; dst_nodes]
      - edge_type      : shape (num_edges,)   —— 每条边的 EdgeType 值
      - num_nodes      : 总节点数 = num_attrs + num_objs + num_train_pairs

    使用方法
    --------
    constructor = HeterogeneousGraphConstructor(attrs, objs, train_pairs)
    graph_data  = constructor.build()    # 返回 dict
    """

    def __init__(
        self,
        attrs       : List[str],
        objs        : List[str],
        train_pairs : List[Tuple[str, str]],
        cache_path  : str = None,
    ):
        self.attrs       = attrs
        self.objs        = objs
        self.train_pairs = train_pairs
        self.cache_path  = cache_path

        # 原语 → 全局节点索引
        self.num_attrs = len(attrs)
        self.num_objs  = len(objs)
        self.num_comps = len(train_pairs)
        self.num_nodes = self.num_attrs + self.num_objs + self.num_comps

        self.attr2nid  = {a: i                         for i, a in enumerate(attrs)}
        self.obj2nid   = {o: i + self.num_attrs         for i, o in enumerate(objs)}
        self.comp2nid  = {p: i + self.num_attrs + self.num_objs
                          for i, p in enumerate(train_pairs)}

    # ────────────────────────────
    # 对外接口
    # ────────────────────────────

    def build(self) -> dict:
        """
        构建图并返回 graph_data 字典。
        若 cache_path 存在则直接读缓存，否则构建后保存。
        """
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[GraphConstructor] 从缓存加载图：{self.cache_path}")
            return torch.load(self.cache_path)

        print(f"[GraphConstructor] 构建异构有向图 ...")
        print(f"  节点数：attr={self.num_attrs}, obj={self.num_objs}, "
              f"comp={self.num_comps}, total={self.num_nodes}")

        src_list  = []
        dst_list  = []
        type_list = []

        # ── 0. Self-loop（所有节点） ──
        for nid in range(self.num_nodes):
            src_list.append(nid);  dst_list.append(nid)
            type_list.append(int(EdgeType.SELF_LOOP))

        # ── 针对每个训练组合对，添加 4 类边 ──
        for (attr, obj), comp_nid in self.comp2nid.items():
            attr_nid = self.attr2nid[attr]
            obj_nid  = self.obj2nid[obj]

            # 1. ATTR → COMP
            src_list.append(attr_nid); dst_list.append(comp_nid)
            type_list.append(int(EdgeType.ATTR2COMP))

            # 2. COMP → ATTR
            src_list.append(comp_nid); dst_list.append(attr_nid)
            type_list.append(int(EdgeType.COMP2ATTR))

            # 3. OBJ → COMP
            src_list.append(obj_nid);  dst_list.append(comp_nid)
            type_list.append(int(EdgeType.OBJ2COMP))

            # 4. COMP → OBJ
            src_list.append(comp_nid); dst_list.append(obj_nid)
            type_list.append(int(EdgeType.COMP2OBJ))

        # ── 针对每个训练组合对，添加 attr ↔ obj 两类边 ──
        # 为避免重复（同一 attr-obj 对可能出现在多个 comp 中），先去重
        attr_obj_pairs_seen = set()
        for (attr, obj) in self.train_pairs:
            if (attr, obj) in attr_obj_pairs_seen:
                continue
            attr_obj_pairs_seen.add((attr, obj))

            attr_nid = self.attr2nid[attr]
            obj_nid  = self.obj2nid[obj]

            # 5. ATTR → OBJ
            src_list.append(attr_nid); dst_list.append(obj_nid)
            type_list.append(int(EdgeType.ATTR2OBJ))

            # 6. OBJ → ATTR
            src_list.append(obj_nid);  dst_list.append(attr_nid)
            type_list.append(int(EdgeType.OBJ2ATTR))

        # ── 构建 Tensor ──
        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )  # (2, E)
        edge_type  = torch.tensor(type_list, dtype=torch.long)  # (E,)

        # ── 节点全局索引 ──
        node_ids_attr = torch.arange(self.num_attrs, dtype=torch.long)
        node_ids_obj  = torch.arange(
            self.num_attrs, self.num_attrs + self.num_objs, dtype=torch.long
        )
        node_ids_comp = torch.arange(
            self.num_attrs + self.num_objs, self.num_nodes, dtype=torch.long
        )

        graph_data = {
            "edge_index"    : edge_index,       # (2, E)
            "edge_type"     : edge_type,        # (E,)
            "num_nodes"     : self.num_nodes,
            "num_attrs"     : self.num_attrs,
            "num_objs"      : self.num_objs,
            "num_comps"     : self.num_comps,
            "num_relations" : NUM_RELATIONS,
            "node_ids_attr" : node_ids_attr,    # (A,)
            "node_ids_obj"  : node_ids_obj,     # (O,)
            "node_ids_comp" : node_ids_comp,    # (C,)
            # 节点映射（供 RGCNEncoder 初始化节点嵌入）
            "attr2nid"      : self.attr2nid,
            "obj2nid"       : self.obj2nid,
            "comp2nid"      : self.comp2nid,
            "attrs"         : self.attrs,
            "objs"          : self.objs,
            "train_pairs"   : self.train_pairs,
        }

        # 打印边统计
        self._print_edge_stats(edge_type)

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(graph_data, self.cache_path)
            print(f"[GraphConstructor] 图已缓存到：{self.cache_path}")

        return graph_data

    # ────────────────────────────
    # 调试工具
    # ────────────────────────────

    def _print_edge_stats(self, edge_type: torch.Tensor):
        """打印各类边的数量统计"""
        print("[GraphConstructor] 边统计：")
        for et in EdgeType:
            cnt = (edge_type == int(et)).sum().item()
            print(f"  {et.name:12s} (type={int(et)}) : {cnt} 条")
        print(f"  {'TOTAL':12s}              : {len(edge_type)} 条")