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

import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from data.data_utils import load_glove_embeddings


class FeasibilityCalibrator:
    """
    Parameters
    ----------
    attrs        : 属性列表
    objs         : 物体列表
    seen_pairs   : 训练可见对列表
    glove_path   : GloVe 词向量文件路径（可选；若为 None，跳过）
    """

    def __init__(
        self,
        attrs      : List[str],
        objs       : List[str],
        seen_pairs : List[Tuple[str, str]],
        glove_path : Optional[str] = None,
    ):
        self.attrs      = attrs
        self.objs       = objs
        self.seen_pairs = seen_pairs
        self.seen_set   = set(seen_pairs)

        # ── 加载 GloVe ──
        self.glove_path = glove_path
        self._feasibility_cache: Dict[Tuple, float] = {}

        if glove_path is not None:
            vocab     = attrs + objs
            self.glove = load_glove_embeddings(glove_path, vocab, dim=300)
            self._precompute_feasibility()
        else:
            print("[Feasibility] 未提供 GloVe 路径，可行性校准已禁用")
            self.glove = None

    # ────────────────────────────
    # 预计算全部未见对的可行性分数
    # ────────────────────────────

    def _precompute_feasibility(self):
        """
        预计算所有 attr×obj 对的可行性分数，缓存到 self._feasibility_cache。
        """
        from itertools import product as iproduct
        import numpy as np

        print("[Feasibility] 预计算可行性分数...")

        # 构建辅助集合
        # applicable_attrs[o] = {a : (a, o) in seen_pairs}
        applicable_attrs: Dict[str, Set[str]] = {o: set() for o in self.objs}
        # applicable_objs[a]  = {o : (a, o) in seen_pairs}
        applicable_objs : Dict[str, Set[str]] = {a: set() for a in self.attrs}

        for (a, o) in self.seen_pairs:
            applicable_attrs[o].add(a)
            applicable_objs[a].add(o)

        def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0
            return float(np.dot(v1, v2) / (n1 * n2))

        for a in self.attrs:
            for o in self.objs:
                if (a, o) in self.seen_set:
                    # 可见对可行性为 1.0
                    self._feasibility_cache[(a, o)] = 1.0
                    continue

                vec_a = self.glove.get(a, np.zeros(300))
                vec_o = self.glove.get(o, np.zeros(300))

                # attr 可行性：在 applicable_attrs[o] 中找与 a 最相似的
                attr_feas = 0.0
                for a_seen in applicable_attrs.get(o, []):
                    sim = _cosine(vec_a, self.glove.get(a_seen, np.zeros(300)))
                    attr_feas = max(attr_feas, sim)

                # obj 可行性：在 applicable_objs[a] 中找与 o 最相似的
                obj_feas = 0.0
                for o_seen in applicable_objs.get(a, []):
                    sim = _cosine(vec_o, self.glove.get(o_seen, np.zeros(300)))
                    obj_feas = max(obj_feas, sim)

                self._feasibility_cache[(a, o)] = (attr_feas + obj_feas) / 2.0

        print(f"[Feasibility] 完成，共 {len(self._feasibility_cache)} 个 pair")

    # ────────────────────────────
    # 对外接口
    # ────────────────────────────

    def get_score(self, attr: str, obj: str) -> float:
        """返回单个 pair 的可行性分数"""
        return self._feasibility_cache.get((attr, obj), 0.0)

    def get_feasibility_mask(
        self,
        pairs    : List[Tuple[str, str]],
        threshold: float,
    ) -> np.ndarray:
        """
        返回可行性 mask，True 表示该 pair 可行性 >= threshold。

        Parameters
        ----------
        pairs     : 候选 pair 列表
        threshold : 可行性阈值

        Returns
        -------
        mask : np.ndarray of bool，shape (len(pairs),)
        """
        if self.glove is None:
            # 没有 GloVe，所有 pair 均视为可行
            return np.ones(len(pairs), dtype=bool)

        return np.array(
            [self._feasibility_cache.get(p, 0.0) >= threshold
             for p in pairs],
            dtype=bool,
        )