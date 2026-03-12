"""
llm_feature_loader.py
---------------------
统一加载两类 LLM 预生成文件：
  1. descriptions_path : {attr obj → str}   节点涌现描述 (mit_qwen.json 格式)
  2. edge_weights_path : {attr obj → float} 边适用性权重 (generate_edge_weights.py 输出)

对外提供：
  - load_node_descriptions(path, node_list, node_type)
      → dict {name: str}  缺失项用 name 本身填充
  - load_edge_weights(path, train_pairs)
      → dict {(attr, obj): float}  缺失项用 0.5 填充
  - load_node_features(path, node_list, clip_text_encoder)
      → torch.Tensor  从 .pt 加载特征，缺失项用 CLIP 实时补齐
"""

import json
import os
import torch  # [修复] 补充缺失的 torch 导入
from typing import Dict, List, Optional, Tuple


def load_node_descriptions(
    descriptions_path : str,
    node_list         : List[str],
    node_type         : str = "primitive",   # "primitive" or "composition"
) -> Dict[str, str]:
    """
    从 JSON 文件加载节点的 LLM 涌现描述。
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
        clean_name = name.replace("_", " ").replace(".", " ").strip()
        if clean_name in raw:
            result[name] = raw[clean_name]
        elif name in raw:
            result[name] = raw[name]
        else:
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


def load_node_features(
    features_path: str,
    node_list: List[str],
    clip_text_encoder, 
    feature_dim: int = 768
) -> torch.Tensor:
    """从 .pt 文件加载预计算的 LLM 节点特征"""
    
    if features_path and os.path.exists(features_path):
        features_dict = torch.load(features_path, map_location="cpu")
        print(f"[LLMLoader] 成功加载离线特征文件：{features_path}")
    else:
        print(f"[LLMLoader] 警告：未找到特征文件 {features_path}，将触发 CLIP 补齐。")
        features_dict = {}

    out_feats = []
    missing_nodes = []
    
    for name in node_list:
        clean_name = name.replace("_", " ").replace(".", " ").strip()
        
        # 匹配字典中的特征
        if clean_name in features_dict:
            out_feats.append(features_dict[clean_name])
        elif name in features_dict:
            out_feats.append(features_dict[name])
        else:
            missing_nodes.append(name)
            
    # 如果有缺失的节点，调用原生的 CLIP text encoder 即时补齐作为 fallback
    if len(missing_nodes) > 0:
        print(f"[LLMLoader] {len(missing_nodes)} 个节点在 .pt 中缺失，使用原始 CLIP 名称特征补齐...")
        with torch.no_grad():
            fallback_feats = clip_text_encoder(missing_nodes).cpu()
            
        # 填补进去
        miss_idx = 0
        final_feats = []
        for name in node_list:
            clean_name = name.replace("_", " ").replace(".", " ").strip()
            if clean_name in features_dict:
                final_feats.append(features_dict[clean_name])
            elif name in features_dict:
                final_feats.append(features_dict[name])
            else:
                final_feats.append(fallback_feats[miss_idx])
                miss_idx += 1
        out_feats = final_feats

    return torch.stack(out_feats, dim=0).float()