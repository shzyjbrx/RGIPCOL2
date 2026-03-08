"""
llm_feature_loader.py
---------------------
统一加载两类 LLM 预生成文件：
  1. descriptions_path : {attr obj → str}   节点涌现描述 (mit_qwen.json 格式)
  2. edge_weights_path : {attr obj → float} 边适用性权重 (generate_edge_weights.py 输出)

对外提供：
  - load_node_descriptions(path, node_list, node_type)
      → dict {name: str}   缺失项用 name 本身填充
  - load_edge_weights(path, train_pairs)
      → dict {(attr, obj): float}  缺失项用 0.5 填充
"""

import json
import os
from typing import Dict, List, Optional, Tuple


def load_node_descriptions(
    descriptions_path : str,
    node_list         : List[str],
    node_type         : str = "primitive",   # "primitive" or "composition"
) -> Dict[str, str]:
    """
    从 JSON 文件加载节点的 LLM 涌现描述。

    JSON 格式（mit_qwen.json）：
        {
            "sliced apple": "A sliced apple shows ...",
            "old city":     "An old city features ...",
            ...
        }

    对于 attr/obj 节点（node_type="primitive"），key 就是节点名；
    对于 comp 节点（node_type="composition"），key 是 "attr obj" 格式。

    Parameters
    ----------
    descriptions_path : JSON 文件路径
    node_list         : 需要查找的节点名称列表
    node_type         : 节点类型标识（用于构建 key）

    Returns
    -------
    dict: {node_name: description_str}  缺失项用节点名本身代替（回退策略）
    """
    raw = {}
    if descriptions_path and os.path.exists(descriptions_path):
        with open(descriptions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"[LLMLoader] 加载节点描述：{descriptions_path}，共 {len(raw)} 条")
    else:
        print(f"[LLMLoader] 节点描述文件未找到：{descriptions_path}，使用空描述回退")

    result = {}
    missing_cnt = 0
    for name in node_list:
        # 尝试直接匹配（key 格式：name 或 "attr obj"）
        clean_name = name.replace("_", " ").replace(".", " ").strip()
        if clean_name in raw:
            result[name] = raw[clean_name]
        elif name in raw:
            result[name] = raw[name]
        else:
            # 回退：使用节点名本身作为描述（CLIP 编码名称时的默认行为）
            result[name] = clean_name
            missing_cnt += 1

    if missing_cnt > 0:
        print(f"[LLMLoader] {missing_cnt}/{len(node_list)} 个节点无LLM描述，使用名称回退")

    return result


def load_edge_weights(
    edge_weights_path: Optional[str],
    train_pairs      : List[Tuple[str, str]],
    default_weight   : float = 0.5,
) -> Dict[Tuple[str, str], float]:
    """
    从 JSON 文件加载 (attr, obj) 边的 LLM 适用性权重。

    JSON 格式（generate_edge_weights.py 输出）：
        {
            "sliced apple": 0.95,
            "broken glass": 0.98,
            "broken air":   0.05,
            ...
        }

    Parameters
    ----------
    edge_weights_path : JSON 文件路径（可为 None）
    train_pairs       : 训练组合对列表
    default_weight    : 文件缺失或 key 缺失时的默认权重

    Returns
    -------
    dict: {(attr, obj): float}
    """
    raw = {}
    if edge_weights_path and os.path.exists(edge_weights_path):
        with open(edge_weights_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"[LLMLoader] 加载边权重：{edge_weights_path}，共 {len(raw)} 条")
    else:
        print(f"[LLMLoader] 边权重文件未找到：{edge_weights_path}，所有边权重设为 {default_weight}")

    result = {}
    missing_cnt = 0
    for (attr, obj) in train_pairs:
        key = f"{attr} {obj}"
        clean_key = key.replace("_", " ").replace(".", " ").strip()
        if clean_key in raw:
            result[(attr, obj)] = float(raw[clean_key])
        elif key in raw:
            result[(attr, obj)] = float(raw[key])
        else:
            result[(attr, obj)] = default_weight
            missing_cnt += 1

    if missing_cnt > 0:
        print(f"[LLMLoader] {missing_cnt}/{len(train_pairs)} 个 pair 无边权重，使用默认 {default_weight}")

    return result