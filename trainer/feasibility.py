"""
feasibility.py
--------------
开放世界 CZSL 可行性校准（Feasibility Calibration）。

来自 CompCos / GIPCOL 的方法：
  对未见组合 (a, o)，计算可行性分数 = (attr_feasibility + obj_feasibility) / 2

  attr_feasibility(a, o) = max_{(a_i, o) ∈ seen} cosine(GloVe(a), GloVe(a_i))
  obj_feasibility(a, o)  = max_{(a, o_j) ∈ seen} cosine(GloVe(o), GloVe(o_j))

使用 GloVe 词向量（300d）计算语义相似度。
"""

import torch
import numpy as np

class FeasibilityCalibrator:
    def __init__(self, feasibility_path: str, dataset):
        print(f"[Feasibility] 正在加载官方预计算文件: {feasibility_path}")
        # 读取官方格式 {'feasibility': 1D Tensor}
        raw_data = torch.load(feasibility_path, map_location='cpu')
        
        if 'feasibility' in raw_data:
            scores_tensor = raw_data['feasibility'].squeeze()
        else:
            scores_tensor = raw_data.squeeze()
            
        self._feasibility_cache = {}
        # 官方代码生成的 tensor 严格按照 dataset.pairs 的顺序排列
        all_pairs = dataset.pairs if hasattr(dataset, 'pairs') else dataset.all_pairs
        
        if len(scores_tensor) == len(all_pairs):
            for idx, pair in enumerate(all_pairs):
                self._feasibility_cache[pair] = scores_tensor[idx].item()
            print(f"[Feasibility] 完美对齐！成功缓存了 {len(self._feasibility_cache)} 个组合的分数。")
        else:
            print(f"[!] 警告：Tensor 长度 ({len(scores_tensor)}) 与候选对总数 ({len(all_pairs)}) 不匹配！")

        # 确保真实训练集组合分数为 1.0
        for pair in dataset.train_pairs:
            self._feasibility_cache[pair] = 1.0

    def get_feasibility_mask(self, pairs, threshold: float) -> np.ndarray:
        return np.array([self._feasibility_cache.get(p, 0.0) >= threshold for p in pairs], dtype=bool)