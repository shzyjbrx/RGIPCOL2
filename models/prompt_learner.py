"""
prompt_learner.py
-----------------
软提示学习模块（Soft Prompt Learner）。

构建 GIPCOL 风格的软提示序列：
    [SOS, θ_1, θ_2, ..., θ_k, â_i, ô_i, EOS]
     ^^^                       ^^^^^^^^^^^
     固定                      GNN/RGCN 更新后的概念嵌入

其中：
  - SOS / EOS    ：CLIP 的特殊起始/终止 token（固定，从 CLIP token embedding 获取）
  - θ_1..θ_k     ：k 个可学习前缀向量（本模块的核心参数）
  - â_i / ô_i   ：RGCN 更新后的 attr / obj 嵌入（由 RGCNEncoder 提供，随梯度流动）

整个序列通过 CLIP 冻结的文本编码器后，取 EOS 位置输出作为组合概念向量 c_i。

注意：CLIP 文本 Transformer 依赖位置编码（positional_embedding），
      本模块在拼接时需要叠加位置编码。
"""

from typing import List, Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLearner(nn.Module):
    """
    可学习软提示模块。

    Parameters
    ----------
    clip_adapter   : CLIPAdapter 实例（用于获取 SOS/EOS embedding、位置编码等）
    prefix_length  : 前缀向量数量 k（默认 3，与 "a photo of" 等长）
    feature_dim    : 特征维度（默认 768，ViT-L/14）
    """

    def __init__(
        self,
        clip_adapter,
        prefix_length : int = 3,
        feature_dim   : int = 768,
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.feature_dim   = feature_dim
        self.clip_adapter  = clip_adapter

        # ── 初始化前缀向量（可学习参数） ──
        # 使用 CLIP 的 token embedding 均值初始化，使其接近合理的语义空间
        with torch.no_grad():
            # "a photo of" 的 token id
            template    = clip.tokenize(["a photo of"]).to(
                next(clip_adapter.clip_model.parameters()).device
            )
            token_embs  = clip_adapter.get_token_embedding(template)  # (1, 77, D)
            # 取前 k 个 token 的嵌入作为初始值（跳过 SOS）
            init_prefix = token_embs[0, 1 : 1 + prefix_length, :]      # (k, D)

        # 以 "a photo of" 前 k token 热启动
        self.prefix_vectors = nn.Parameter(init_prefix.clone())  # (k, D)

        # ── 预计算 SOS / EOS token 嵌入（固定，不参与梯度） ──
        # SOS = token id 49406，EOS = token id 49407（CLIP BPE 标准）
        with torch.no_grad():
            device   = next(clip_adapter.clip_model.parameters()).device
            sos_id   = torch.tensor([[49406]], device=device)  # SOS
            eos_id   = torch.tensor([[49407]], device=device)  # EOS
            sos_emb  = clip_adapter.get_token_embedding(sos_id)[0, 0, :]  # (D,)
            eos_emb  = clip_adapter.get_token_embedding(eos_id)[0, 0, :]  # (D,)

        self.register_buffer("sos_emb", sos_emb.float())  # (D,)
        self.register_buffer("eos_emb", eos_emb.float())  # (D,)

        # ── 序列总长度 ──
        # [SOS] + [k prefix] + [attr] + [obj] + [EOS]  = k + 4
        self.seq_len = 1 + prefix_length + 1 + 1 + 1   # = k + 4
        # EOS 的位置索引（0-based）
        self.eos_pos = self.seq_len - 1

    # ────────────────────────────
    # 构建软提示序列
    # ────────────────────────────

    def build_prompt_embeddings(
        self,
        attr_embs : torch.Tensor,  # (num_pairs, D)  — RGCN 更新后的 attr 嵌入
        obj_embs  : torch.Tensor,  # (num_pairs, D)  — RGCN 更新后的 obj  嵌入
    ) -> torch.Tensor:
        """
        为每个 (attr, obj) 对构建完整的软提示序列嵌入。

        Parameters
        ----------
        attr_embs : (P, D)  — P 个 pair 对应的 attr 嵌入
        obj_embs  : (P, D)  — P 个 pair 对应的 obj  嵌入

        Returns
        -------
        prompt_embs : (P, seq_len, D)  — 完整提示序列嵌入（含位置编码）
        """
        P, D = attr_embs.size()
        device = attr_embs.device

        # ── 拼接序列 ──
        # SOS:    (1, D) → (P, 1, D)
        sos  = self.sos_emb.unsqueeze(0).unsqueeze(0).expand(P, 1, D)
        # prefix: (k, D) → (P, k, D)
        prefix = self.prefix_vectors.unsqueeze(0).expand(P, -1, -1)
        # attr:   (P, D) → (P, 1, D)
        attr = attr_embs.unsqueeze(1)
        # obj:    (P, D) → (P, 1, D)
        obj  = obj_embs.unsqueeze(1)
        # EOS:    (1, D) → (P, 1, D)
        eos  = self.eos_emb.unsqueeze(0).unsqueeze(0).expand(P, 1, D)

        # 序列：[SOS, θ_1..θ_k, attr, obj, EOS]
        prompt_embs = torch.cat([sos, prefix, attr, obj, eos], dim=1)  # (P, seq_len, D)

        # ── 叠加 CLIP 位置编码 ──
        pos_emb = self.clip_adapter.positional_embedding[: self.seq_len, :]  # (seq_len, D)
        prompt_embs = prompt_embs + pos_emb.unsqueeze(0)  # (P, seq_len, D)

        return prompt_embs

    # ────────────────────────────
    # 前向传播：生成组合概念向量
    # ────────────────────────────

    def forward(
        self,
        attr_embs : torch.Tensor,  # (P, D)
        obj_embs  : torch.Tensor,  # (P, D)
    ) -> torch.Tensor:
        """
        构建提示序列 → 送入 CLIP 文本 Transformer → 取 EOS 位置输出。

        Returns
        -------
        concept_vecs : (P, feature_dim)  L2 归一化的组合概念向量
        """
        # 构建软提示嵌入序列
        prompt_embs = self.build_prompt_embeddings(attr_embs, obj_embs)
        # (P, seq_len, D)

        # 送入冻结的 CLIP 文本 Transformer
        # encode_text_tokens 期望 (B, seq_len, D)
        concept_vecs = self.clip_adapter.encode_text_tokens(prompt_embs)
        # (P, feature_dim)，已 L2 归一化

        return concept_vecs

    def extra_repr(self) -> str:
        return (
            f"prefix_length={self.prefix_length}, "
            f"seq_len={self.seq_len}, "
            f"feature_dim={self.feature_dim}"
        )