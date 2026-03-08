"""
data_utils.py
-------------
数据处理相关工具函数：
  - GloVe 词向量加载（用于开放世界可行性校准）
  - 配置文件加载（支持 _base_ 继承）
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────
# 配置加载（支持 _base_ 继承）
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """
    加载 yaml 配置文件，支持通过 _base_ 字段继承基础配置。
    子配置的字段会覆盖父配置的同名字段。

    Parameters
    ----------
    config_path : yaml 文件路径

    Returns
    -------
    合并后的配置字典
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    # 处理 _base_ 继承
    if "_base_" in cfg:
        base_name = cfg.pop("_base_")
        base_path = os.path.join(config_dir, base_name)
        base_cfg  = load_config(base_path)    # 递归支持多级继承
        base_cfg.update(cfg)                  # 子配置覆盖父配置
        cfg = base_cfg

    return cfg


# ──────────────────────────────────────────────────────────────
# GloVe 加载
# ──────────────────────────────────────────────────────────────

def load_glove_embeddings(
    glove_path: str,
    vocab: List[str],
    dim: int = 300,
) -> Dict[str, np.ndarray]:
    """
    从 GloVe 文本文件中加载指定词汇的词向量。

    Parameters
    ----------
    glove_path : GloVe 文件路径，例如 "glove.6B.300d.txt"
    vocab      : 需要加载的词汇列表（attr/obj 名称）
    dim        : GloVe 维度，默认 300

    Returns
    -------
    dict: {word: np.ndarray of shape (dim,)}
    未找到的词将使用零向量填充，并打印警告。
    """
    vocab_set    = set(vocab)
    embeddings   = {}
    missing      = []

    print(f"[GloVe] 正在从 {glove_path} 加载词向量...")

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab_set:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == dim:
                    embeddings[word] = vec

    # 对多词短语（如 "sliced_apple"），尝试取各词均值
    for word in vocab_set:
        if word not in embeddings:
            sub_words = word.replace("_", " ").replace(".", " ").split()
            sub_vecs  = [embeddings[w] for w in sub_words if w in embeddings]
            if sub_vecs:
                embeddings[word] = np.mean(sub_vecs, axis=0)
            else:
                missing.append(word)
                embeddings[word] = np.zeros(dim, dtype=np.float32)

    if missing:
        print(f"[GloVe] 警告：以下 {len(missing)} 个词未找到向量，使用零向量填充：")
        print("  ", missing[:20], "..." if len(missing) > 20 else "")

    print(f"[GloVe] 成功加载 {len(embeddings) - len(missing)}/{len(vocab)} 个词向量")
    return embeddings


# ──────────────────────────────────────────────────────────────
# 数据集元信息打印
# ──────────────────────────────────────────────────────────────

def print_dataset_stats(dataset) -> None:
    """打印数据集统计信息"""
    print("=" * 50)
    print(f"Dataset : {dataset.data_dir}")
    print(f"Split   : {dataset.split}")
    print(f"#Attrs  : {dataset.num_attrs}")
    print(f"#Objs   : {dataset.num_objs}")
    print(f"#TrainPairs  : {dataset.num_train_pairs}")
    print(f"#SeenPairs   : {len(dataset.seen_pairs)}")
    print(f"#UnseenPairs : {len(dataset.unseen_pairs)}")
    print(f"#AllPairs    : {len(dataset.all_pairs)}")
    print(f"#Samples     : {len(dataset)}")
    print("=" * 50)