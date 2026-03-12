"""
feasibility.py
--------------
开放世界 CZSL 双阶段可行性过滤器（论文 3.2.6 节）

Stage-1：加载官方预计算 feasibility .pt 文件进行粗筛
    C₁ = {(a, o) | score_pt(a, o) >= θ₁}

Stage-2：LLM 物理合理性精筛
    - 优先从 JSON 缓存读取已有分数
    - 对 C₁ 中缺少分数的 pair，调用本地 Qwen 模型在线生成并写回缓存
    C_final = {(a, o) ∈ C₁ | s_LLM(a, o) >= θ₂}   [公式 3-12]

设计要点：
  - 训练集（seen pair）无条件保留，不受任何阈值约束
  - LLM 仅在有未缓存 pair 时才懒加载，生成完毕后立即释放显存
  - 结果实时写回 JSON 缓存，下次启动直接复用，无需重复生成
  - 旧版 FeasibilityCalibrator 保留，供消融实验对比
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
# 双阶段可行性过滤器
# ──────────────────────────────────────────────────────────────

class TwoStageFeasibilityCalibrator:
    """
    双阶段可行性过滤器。

    Parameters
    ----------
    dataset               : CompositionDataset（需含 train_pairs / all_pairs）
    feasibility_path      : 官方预计算 .pt 文件路径（Stage-1 粗筛来源）
    llm_edge_weights_path : LLM 物理合理性分数 JSON 缓存路径（Stage-2）
                            不存在时自动创建；已有分数直接复用，缺失的在线生成
    theta1                : Stage-1 粗筛阈值（建议偏松，保留足够候选）
    theta2                : Stage-2 LLM 精筛阈值
    llm_model_id          : 本地 Qwen 模型路径或 HuggingFace ID
    max_retries           : LLM 生成失败时的最大重试次数
    """

    def __init__(
        self,
        dataset,
        feasibility_path      : str,
        llm_edge_weights_path : str,
        theta1                : float = 0.4,
        theta2                : float = 0.5,
        llm_model_id          : str   = "Qwen/Qwen2.5-7B-Instruct",
        max_retries           : int   = 3,
    ):
        self.theta1               = theta1
        self.theta2               = theta2
        self.max_retries          = max_retries
        self.llm_model_id         = llm_model_id
        self.llm_weights_path     = llm_edge_weights_path
        self.train_pairs_set      = set(map(tuple, dataset.train_pairs))

        # ── Stage-1：加载官方预计算 .pt 分数 ──
        self._pt_scores: Dict[Tuple, float] = self._load_pt_scores(
            feasibility_path, dataset
        )

        # ── Stage-2：加载已有 LLM 分数缓存 ──
        self._llm_scores: Dict[str, float] = self._load_llm_cache(
            llm_edge_weights_path
        )

        # LLM 懒加载：仅在需要时才载入
        self._llm_tokenizer = None
        self._llm_model     = None

        print(
            f"[TwoStage] 初始化完成 | "
            f"θ₁={theta1} (.pt粗筛) | θ₂={theta2} (LLM精筛) | "
            f"已缓存 LLM 分数：{len(self._llm_scores)} 条"
        )

    # ──────────────────────────────────────────────
    # Stage-1：加载 .pt 分数
    # ──────────────────────────────────────────────

    def _load_pt_scores(
        self, feasibility_path: str, dataset
    ) -> Dict[Tuple, float]:
        """从官方预计算 .pt 文件加载分数，按 all_pairs 顺序对齐。"""
        print(f"[TwoStage Stage-1] 加载预计算文件：{feasibility_path}")
        raw = torch.load(feasibility_path, map_location="cpu")

        scores_t = (
            raw["feasibility"].squeeze()
            if "feasibility" in raw
            else raw.squeeze()
        )
        all_pairs = (
            dataset.pairs if hasattr(dataset, "pairs") else dataset.all_pairs
        )

        if len(scores_t) != len(all_pairs):
            print(
                f"[TwoStage] ⚠ .pt 长度 ({len(scores_t)}) 与 "
                f"all_pairs ({len(all_pairs)}) 不匹配，"
                f"Stage-1 将对所有 pair 返回 1.0（不过滤）"
            )
            return {tuple(p): 1.0 for p in all_pairs}

        cache = {tuple(p): scores_t[i].item() for i, p in enumerate(all_pairs)}
        # 训练集分数强制设为 1.0，确保 seen pair 始终通过粗筛
        for p in dataset.train_pairs:
            cache[tuple(p)] = 1.0

        print(
            f"[TwoStage Stage-1] 加载完成，共 {len(cache)} 个 pair 的预计算分数"
        )
        return cache

    # ──────────────────────────────────────────────
    # Stage-2：加载 / 保存 LLM 缓存
    # ──────────────────────────────────────────────

    def _load_llm_cache(self, path: str) -> Dict[str, float]:
        """加载 LLM 分数 JSON 缓存（文件不存在则返回空字典）。"""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                    # 统一规范化 key（下划线/点 → 空格）
                    return {
                        k.replace("_", " ").replace(".", " ").strip(): float(v)
                        for k, v in raw.items()
                    }
                except json.JSONDecodeError:
                    print(f"[TwoStage] ⚠ JSON 缓存解析失败，从空缓存开始")
        return {}

    def _save_llm_cache(self):
        """将当前 LLM 分数写回 JSON 缓存文件。"""
        if not self.llm_weights_path:
            return
        os.makedirs(
            os.path.dirname(os.path.abspath(self.llm_weights_path)),
            exist_ok=True,
        )
        with open(self.llm_weights_path, "w", encoding="utf-8") as f:
            json.dump(self._llm_scores, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    # LLM 懒加载与在线打分
    # ──────────────────────────────────────────────

    def _ensure_llm_loaded(self):
        """仅当有未缓存的 pair 需要打分时，才加载 LLM 模型。"""
        if self._llm_model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "[TwoStage] 需要 transformers 包：pip install transformers"
            )
        print(f"[TwoStage Stage-2] 懒加载 LLM：{self.llm_model_id}")
        self._llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._llm_model.eval()
        print(f"[TwoStage Stage-2] LLM 加载完成")

    def _release_llm(self):
        """打分完毕后释放 LLM 显存。"""
        if self._llm_model is not None:
            del self._llm_model
            del self._llm_tokenizer
            self._llm_model     = None
            self._llm_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[TwoStage Stage-2] LLM 显存已释放")

    @staticmethod
    def _clean_name(name: str) -> str:
        return name.replace("_", " ").replace(".", " ").strip()

    def _build_llm_prompt(self, attr: str, obj: str) -> list:
        """构建物理合理性评分 prompt（与 generate_edge_weights.py 格式一致）。"""
        attr_c = self._clean_name(attr)
        obj_c  = self._clean_name(obj)
        system = (
            "You are a concise physical reasoning assistant. "
            "You only output a single integer from 0 to 10. "
            "No explanation, no punctuation, just the number."
        )
        user = (
            f"Rate the physical applicability of the attribute '{attr_c}' "
            f"to the object '{obj_c}' on a scale from 0 to 10.\n"
            f"0 = physically impossible (e.g., 'broken air'), "
            f"10 = highly natural (e.g., 'broken glass').\n"
            f"Output only a single integer."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    @staticmethod
    def _parse_score(response: str) -> float:
        nums = re.findall(r"\d+", response.strip())
        if not nums:
            return 0.5
        return max(0, min(10, int(nums[0]))) / 10.0

    @staticmethod
    def _contains_invalid(response: str) -> bool:
        if re.search(r"[\u4e00-\u9fff]", response):
            return True
        if len(response.strip()) > 10:
            return True
        return False

    def _score_one_pair(self, attr: str, obj: str) -> float:
        """调用 LLM 对单个 (attr, obj) 打分，含重试机制。"""
        messages = self._build_llm_prompt(attr, obj)
        text = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._llm_tokenizer(
            [text], return_tensors="pt"
        ).to(self._llm_model.device)

        for attempt in range(self.max_retries):
            with torch.no_grad():
                gen_ids = self._llm_model.generate(
                    **model_inputs,
                    max_new_tokens = 8,
                    temperature    = 0.1 + attempt * 0.2,
                    top_p          = 0.9,
                    do_sample      = True,
                    pad_token_id   = self._llm_tokenizer.eos_token_id,
                )
            out_ids  = [
                o[len(i):]
                for i, o in zip(model_inputs.input_ids, gen_ids)
            ]
            response = self._llm_tokenizer.batch_decode(
                out_ids, skip_special_tokens=True
            )[0].strip()

            if not self._contains_invalid(response):
                return self._parse_score(response)

        # 所有重试失败，保守返回 0.5
        print(f"[TwoStage] ⚠ ({attr}, {obj}) LLM 打分失败，使用默认 0.5")
        return 0.5

    def _batch_generate_llm_scores(
        self, pairs_to_score: List[Tuple[str, str]]
    ) -> None:
        """
        对缺少 LLM 分数的 pair 批量在线生成并写入缓存。
        每 50 条自动保存一次，防止中断丢失进度。
        """
        if not pairs_to_score:
            return

        print(
            f"[TwoStage Stage-2] 需要在线生成 {len(pairs_to_score)} 个 pair 的 LLM 分数"
        )
        self._ensure_llm_loaded()

        try:
            from tqdm import tqdm
            iterator = tqdm(pairs_to_score, desc="LLM scoring")
        except ImportError:
            iterator = pairs_to_score

        for cnt, (attr, obj) in enumerate(iterator, start=1):
            key   = f"{self._clean_name(attr)} {self._clean_name(obj)}"
            score = self._score_one_pair(attr, obj)
            self._llm_scores[key] = score

            if cnt % 50 == 0:
                self._save_llm_cache()
                print(
                    f"[TwoStage Stage-2] 自动保存进度 "
                    f"({cnt}/{len(pairs_to_score)})"
                )

        # 最终完整保存
        self._save_llm_cache()
        print(
            f"[TwoStage Stage-2] LLM 分数生成完毕，"
            f"累计缓存 {len(self._llm_scores)} 条 → {self.llm_weights_path}"
        )

        # 释放显存供后续推理使用
        self._release_llm()

    # ──────────────────────────────────────────────
    # 主接口：生成双阶段过滤掩码
    # ──────────────────────────────────────────────

    def get_feasibility_mask(
        self,
        pairs  : List[Tuple[str, str]],
        theta1 : Optional[float] = None,
        theta2 : Optional[float] = None,
    ) -> np.ndarray:
        """
        对候选 pair 列表执行双阶段过滤，返回布尔掩码。

        Parameters
        ----------
        pairs  : 候选 (attr, obj) 对列表（通常为 dataset.all_pairs）
        theta1 : 临时覆盖 Stage-1 阈值（None 使用构造时设定的值）
        theta2 : 临时覆盖 Stage-2 阈值（None 使用构造时设定的值）

        Returns
        -------
        mask : np.ndarray[bool], shape (len(pairs),)
               True = 通过过滤（可行组合），False = 被剔除
        """
        t1 = theta1 if theta1 is not None else self.theta1
        t2 = theta2 if theta2 is not None else self.theta2

        n    = len(pairs)
        mask = np.zeros(n, dtype=bool)

        # ── 训练集（seen pair）无条件保留 ──
        seen_mask = np.array(
            [tuple(p) in self.train_pairs_set for p in pairs], dtype=bool
        )
        mask[seen_mask] = True

        # ── 仅对 unseen pair 执行过滤 ──
        unseen_indices = np.where(~seen_mask)[0]
        if len(unseen_indices) == 0:
            return mask
        unseen_pairs = [pairs[i] for i in unseen_indices]
        total_unseen = len(unseen_pairs)

        # ────────────────────────────
        # Stage-1：.pt 文件粗筛
        # ────────────────────────────
        pt_scores   = np.array(
            [self._pt_scores.get(tuple(p), 0.0) for p in unseen_pairs],
            dtype=np.float32,
        )
        s1_pass         = pt_scores >= t1
        s1_pass_cnt     = int(s1_pass.sum())
        s1_global_idx   = unseen_indices[s1_pass]   # 在 pairs 中的全局下标
        s1_pairs        = [pairs[i] for i in s1_global_idx]

        print(
            f"[TwoStage Stage-1] θ₁={t1:.3f} | "
            f"unseen pair: {total_unseen} → 通过: {s1_pass_cnt} "
            f"({s1_pass_cnt / max(total_unseen, 1) * 100:.1f}%)"
        )

        # ────────────────────────────
        # Stage-2：LLM 物理合理性精筛
        # ────────────────────────────
        # 找出 Stage-1 通过但缺少 LLM 分数的 pair，在线生成
        missing_pairs = [
            (a, o) for a, o in s1_pairs
            if f"{self._clean_name(a)} {self._clean_name(o)}"
               not in self._llm_scores
        ]
        if missing_pairs:
            self._batch_generate_llm_scores(missing_pairs)

        # 读取 LLM 分数并应用 θ₂
        llm_scores_arr = np.array(
            [
                self._llm_scores.get(
                    f"{self._clean_name(a)} {self._clean_name(o)}", 1.0
                )
                for a, o in s1_pairs
            ],
            dtype=np.float32,
        )
        s2_pass         = llm_scores_arr >= t2
        s2_pass_cnt     = int(s2_pass.sum())
        final_global_idx = s1_global_idx[s2_pass]
        mask[final_global_idx] = True

        print(
            f"[TwoStage Stage-2] θ₂={t2:.3f} | "
            f"Stage-1 通过: {s1_pass_cnt} → 最终通过: {s2_pass_cnt} "
            f"({s2_pass_cnt / max(total_unseen, 1) * 100:.1f}% of unseen)"
        )

        return mask

    # ──────────────────────────────────────────────
    # 辅助工具：θ₁ 扫描
    # ──────────────────────────────────────────────

    def scan_theta1(
        self,
        pairs        : List[Tuple[str, str]],
        theta1_range : Optional[np.ndarray] = None,
    ) -> Dict[float, int]:
        """扫描不同 θ₁ 下通过 Stage-1 的未见 pair 数量，辅助阈值选择。"""
        if theta1_range is None:
            theta1_range = np.arange(0.1, 0.9, 0.05)

        seen_mask    = np.array(
            [tuple(p) in self.train_pairs_set for p in pairs], dtype=bool
        )
        unseen_pairs = [pairs[i] for i in np.where(~seen_mask)[0]]
        pt_scores    = np.array(
            [self._pt_scores.get(tuple(p), 0.0) for p in unseen_pairs],
            dtype=np.float32,
        )

        results = {}
        print("[TwoStage] θ₁ 扫描结果（unseen pair 通过数）：")
        for t in theta1_range:
            cnt = int((pt_scores >= t).sum())
            results[float(t)] = cnt
            print(
                f"  θ₁={t:.2f} → {cnt}/{len(unseen_pairs)} "
                f"({cnt / max(len(unseen_pairs), 1) * 100:.1f}%)"
            )
        return results


# ──────────────────────────────────────────────────────────────
# 旧版兼容：单阶段官方预计算（消融实验用）
# ──────────────────────────────────────────────────────────────

class FeasibilityCalibrator:
    """
    仅使用官方预计算 .pt 文件的单阶段过滤器，用于消融对比。
    """

    def __init__(
        self,
        feasibility_path : str,
        dataset,
        threshold        : float = 0.5,
    ):
        self.threshold = threshold
        print(f"[Feasibility] 加载官方预计算文件：{feasibility_path}")
        raw      = torch.load(feasibility_path, map_location="cpu")
        scores_t = (
            raw["feasibility"].squeeze()
            if "feasibility" in raw
            else raw.squeeze()
        )
        all_pairs = (
            dataset.pairs if hasattr(dataset, "pairs") else dataset.all_pairs
        )

        self._cache: Dict[Tuple, float] = {}
        if len(scores_t) == len(all_pairs):
            for idx, p in enumerate(all_pairs):
                self._cache[tuple(p)] = scores_t[idx].item()
            print(f"[Feasibility] 缓存了 {len(self._cache)} 个分数")
        else:
            print(
                f"[!] 长度不匹配（{len(scores_t)} vs {len(all_pairs)}），"
                f"所有分数设为 0.0"
            )
        for p in dataset.train_pairs:
            self._cache[tuple(p)] = 1.0

    def get_feasibility_mask(
        self,
        pairs     : List[Tuple[str, str]],
        threshold : Optional[float] = None,
    ) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        return np.array(
            [self._cache.get(tuple(p), 0.0) >= t for p in pairs],
            dtype=bool,
        )


# ──────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────

def build_feasibility_calibrator(cfg: dict, dataset):
    """
    根据配置自动构建可行性校准器。

    选择逻辑：
      use_two_stage_filter=true  且 feasibility_path 有效
          → TwoStageFeasibilityCalibrator（pt粗筛 + LLM精筛）
      use_two_stage_filter=false 且 feasibility_path 有效
          → FeasibilityCalibrator（单阶段，消融对比）
      两者都不满足
          → None（不过滤）
    """
    feas_path = cfg.get("feasibility_path", None)
    use_two   = cfg.get("use_two_stage_filter", False)

    if use_two:
        if not feas_path or not os.path.exists(str(feas_path)):
            print(
                f"[Feasibility] ⚠ use_two_stage_filter=True 但 "
                f"feasibility_path 无效（{feas_path}），"
                f"无法执行 Stage-1 粗筛，回退到无过滤模式"
            )
            return None

        # 自动推断 LLM 缓存路径
        llm_weights = cfg.get("llm_edge_weights_path", None)
        if not llm_weights:
            dataset_name = cfg.get("dataset", "dataset")
            llm_weights  = f"./LLM/{dataset_name}_edge_weights.json"
            print(
                f"[Feasibility] llm_edge_weights_path 未配置，"
                f"将自动使用：{llm_weights}"
            )

        print(f"[Feasibility] 使用双阶段过滤器（TwoStage）")
        return TwoStageFeasibilityCalibrator(
            dataset               = dataset,
            feasibility_path      = feas_path,
            llm_edge_weights_path = llm_weights,
            theta1                = cfg.get("feasibility_theta1", 0.4),
            theta2                = cfg.get("feasibility_theta2", 0.5),
            llm_model_id          = cfg.get(
                "llm_model_id", "Qwen/Qwen2.5-7B-Instruct"
            ),
            max_retries           = cfg.get("llm_max_retries", 3),
        )

    if feas_path and os.path.exists(str(feas_path)):
        print(f"[Feasibility] 使用官方预计算文件（单阶段）：{feas_path}")
        return FeasibilityCalibrator(
            feasibility_path = feas_path,
            dataset          = dataset,
            threshold        = cfg.get("feasibility_threshold", 0.5),
        )

    print(f"[Feasibility] 未找到可行性文件，开放世界测试不进行过滤")
    return None