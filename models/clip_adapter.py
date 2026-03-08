"""
clip_adapter.py
---------------
CLIP 模型封装器。

职责：
  1. 从本地路径加载预训练 CLIP（ViT-L/14）
  2. 冻结全部 CLIP 参数
  3. 暴露以下接口：
       - encode_image(images)       → 归一化图像特征向量
       - encode_text(token_ids)     → 归一化文本特征向量
       - encode_words(word_list)    → 将词语列表编码为嵌入向量（用于节点初始化）
       - tokenize(texts)            → 文本分词
       - logit_scale               → CLIP 温度参数 exp(log_scale)
"""

from typing import List, Optional, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPAdapter(nn.Module):
    """
    冻结的 CLIP 模型封装。

    Parameters
    ----------
    model_name  : CLIP 模型名称，e.g. "ViT-L/14"
    model_path  : 本地权重文件路径（.pt 文件）
    device      : 运行设备
    """

    def __init__(
        self,
        model_name : str = "ViT-L/14",
        model_path : str = None,
        device     : Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.device     = torch.device(device)
        self.model_name = model_name

        # ── 加载 CLIP ──
        print(f"[CLIPAdapter] 加载 CLIP 模型：{model_name}")
        if model_path is not None:
            print(f"[CLIPAdapter] 使用本地权重：{model_path}")
            # clip.load 支持直接传入本地路径
            self.clip_model, self.preprocess = clip.load(
                model_path,
                device=self.device,
                jit=False,
            )
        else:
            self.clip_model, self.preprocess = clip.load(
                model_name,
                device=self.device,
                jit=False,
            )

        # ── 冻结全部 CLIP 参数 ──
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # ── 特征维度 ──
        # ViT-L/14 的文本和图像特征维度均为 768
        self.feature_dim = self.clip_model.text_projection.shape[1] \
            if hasattr(self.clip_model, "text_projection") \
            else 768

        print(f"[CLIPAdapter] 加载完成，feature_dim={self.feature_dim}")
        print(f"[CLIPAdapter] 参数已全部冻结（requires_grad=False）")

    # ────────────────────────────
    # 图像编码
    # ────────────────────────────

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像，返回 L2 归一化特征向量。

        Parameters
        ----------
        images : (B, 3, H, W)  预处理后的图像 Tensor

        Returns
        -------
        (B, feature_dim)  归一化特征
        """
        feats = self.clip_model.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    # ────────────────────────────
    # 文本编码（软提示 token 序列）
    # ────────────────────────────

    # def encode_text_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
    #     """
    #     将已经嵌入为向量的 token 序列送入 CLIP 文本 Transformer。
    #     这是 soft prompting 的核心：绕过 token embedding 层，直接传入嵌入向量。

    #     CLIP 文本编码器（ViT 风格 Transformer）期望输入为：
    #       (sequence_length, batch_size, embed_dim)

    #     Parameters
    #     ----------
    #     token_embeddings : (B, seq_len, D)  软提示序列向量

    #     Returns
    #     -------
    #     (B, feature_dim)  归一化组合概念向量
    #     """
    #     # 转置为 (seq_len, B, D)
    #     x = token_embeddings.permute(1, 0, 2)

    #     # 将输入张量 x 的类型强转为 CLIP 模型的类型 (通常为 float16)
    #     x = x.to(self.clip_model.dtype)

    #     # 通过 CLIP Transformer（不重新过 token_embedding）
    #     x = self.clip_model.transformer(x)

    #     # 转回 (B, seq_len, D)
    #     x = x.permute(1, 0, 2)

    #     # 取 EOS 位置（CLIP 约定：最后一个非 pad token）
    #     # CLIP 使用 argmax 找 EOS：token_ids 中最大值位置
    #     # 此处软提示没有 token_ids，约定取最后一位
    #     eos_feat = x[:, -1, :]  # (B, D)

    #     # 过 CLIP 的 text_projection
    #     if hasattr(self.clip_model, "text_projection") and \
    #             self.clip_model.text_projection is not None:
    #         eos_feat = eos_feat @ self.clip_model.text_projection

    #     return F.normalize(eos_feat.float(), dim=-1)
    def encode_text_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        将已经嵌入为向量的 token 序列送入 CLIP 文本 Transformer。
        
        Parameters
        ----------
        token_embeddings : (B, seq_len, D)  软提示序列向量

        Returns
        -------
        (B, feature_dim)  归一化组合概念向量
        """
        # 1. 转换数据类型，并保持在 (Batch, Seq_len, Dim) 维度下进行操作
        x = token_embeddings.to(self.clip_model.dtype)
        batch_size, original_seq_len, embed_dim = x.shape
        context_length = self.clip_model.context_length  # 通常是 77
        
        # 2. Padding 补齐到 77
        if original_seq_len < context_length:
            # 构造全 0 的 padding 张量: [batch_size, 77 - seq_len, embed_dim]
            padding = torch.zeros(
                batch_size, 
                context_length - original_seq_len, 
                embed_dim, 
                dtype=x.dtype, 
                device=x.device
            )
            # 拼接到 x 的末尾，x 变为 [batch_size, 77, embed_dim]
            x = torch.cat([x, padding], dim=1)
        elif original_seq_len > context_length:
            # 防止超长，进行截断
            x = x[:, :context_length, :]
            original_seq_len = context_length
            
        # 3. 加上绝对位置编码 (x 是 B, 77, D，可以直接与 77, D 的 positional_embedding 广播相加)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)

        # 4. 转置为 CLIP Transformer 需要的维度: (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # 5. 通过 CLIP Transformer
        x = self.clip_model.transformer(x)

        # 6. 再转回 (batch_size, seq_len, embed_dim) 方便后续提取特征
        x = x.permute(1, 0, 2)

        # 7. 经过 CLIP 最终的 LayerNorm（极其重要，不要漏掉）
        if hasattr(self.clip_model, "ln_final"):
            x = self.clip_model.ln_final(x)

        # 8. 提取真正的 EOS 特征
        # 真正的 EOS 存在于原始序列的最后一个有效位置，即 original_seq_len - 1
        # 使用 torch.arange 取出每个 Batch 对应位置的向量
        eos_feat = x[torch.arange(batch_size, device=x.device), original_seq_len - 1, :]

        # 9. 过 CLIP 的 text_projection 投射到公共多模态空间
        if hasattr(self.clip_model, "text_projection") and self.clip_model.text_projection is not None:
            eos_feat = eos_feat @ self.clip_model.text_projection

        # 10. L2 归一化返回
        return F.normalize(eos_feat.float(), dim=-1)

    # ────────────────────────────
    # 词语编码（节点初始化用）
    # ────────────────────────────

    @torch.no_grad()
    def encode_words(self, word_list: List[str]) -> torch.Tensor:
        """
        将词语列表编码为 CLIP 文本嵌入向量。
        使用标准 CLIP 文本编码（tokenize → encode_text）。

        Parameters
        ----------
        word_list : 词语列表，e.g. ["sliced", "apple", ...]

        Returns
        -------
        (len(word_list), feature_dim)  归一化词嵌入
        """
        # CLIP tokenize 支持批处理
        token_ids = clip.tokenize(word_list, truncate=True).to(self.device)
        feats     = self.clip_model.encode_text(token_ids)
        return F.normalize(feats.float(), dim=-1)

    # ────────────────────────────
    # Token Embedding 层（软提示构建用）
    # ────────────────────────────

    def get_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        获取 CLIP token embedding 层的输出（word embedding，不含位置编码）。

        Parameters
        ----------
        token_ids : (B, seq_len)  整数 token id

        Returns
        -------
        (B, seq_len, D)  token 嵌入向量
        """
        return self.clip_model.token_embedding(token_ids).float()

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """将文本列表分词，返回 token id Tensor"""
        return clip.tokenize(texts, truncate=True).to(self.device)

    # ────────────────────────────
    # 温度参数
    # ────────────────────────────

    @property
    def logit_scale(self) -> torch.Tensor:
        """返回 CLIP 的 logit_scale（exp 后即为温度的倒数 1/τ）"""
        return self.clip_model.logit_scale.exp().float()

    # ────────────────────────────
    # 位置编码（供 PromptLearner 使用）
    # ────────────────────────────

    @property
    def positional_embedding(self) -> torch.Tensor:
        return self.clip_model.positional_embedding.float()

    @property
    def ln_final(self):
        return self.clip_model.ln_final

    def extra_repr(self) -> str:
        return f"model={self.model_name}, feature_dim={self.feature_dim}"