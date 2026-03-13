"""
visualize/attention_weights.py
-------------------------------
软提示注意力权重可视化（论文图3-X）

可视化目标：
  对比同一组合（如 "sliced apple"）在两种编码方式下，
  CLIP 文本 Transformer 最后 N 层对软提示序列的注意力权重：

    方式 A（无图注入）：序列 = [SOS, θ₁..θₖ, â_clip, ô_clip, EOS]
                         â/ô 使用 CLIP 原始词汇嵌入
    方式 B（有图注入）：序列 = [SOS, θ₁..θₖ, â_rgcn, ô_rgcn, EOS]
                         â/ô 使用 RGCN 传播更新后的嵌入

输出图：
  1. attention_heatmap_<comp>.png   — 指定组合的逐层注意力热力图
  2. attention_eos_barplot.png      — EOS 位置对各 token 的注意力均值对比柱状图
  3. attention_diff_heatmap.png     — 有/无图注入的注意力差异热力图

使用方法：
    python visualize/attention_weights.py \
        --config   configs/mit_states_config.yaml \
        --checkpoint saved_models/mit-states/xxx/best_model.pt \
        --comps "sliced apple" "wet dog" "broken glass" \
        --save_dir ./visualize/outputs/attention

依赖：
    pip install matplotlib seaborn scikit-learn
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils import load_config
from data.dataset import get_dataloader
from models.rgipcol import RGIPCOL
from utils.checkpoint import load_checkpoint


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Soft Prompt Attention Weight Visualization")
    p.add_argument("--config",      type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--comps",       type=str, nargs="+",
                   default=["sliced apple", "wet dog", "broken glass"],
                   help="需要可视化的组合，格式为 'attr obj'")
    p.add_argument("--top_layers",  type=int, default=4,
                   help="可视化最后 N 层的注意力权重（默认4层）")
    p.add_argument("--save_dir",    type=str,
                   default="./visualize/outputs/attention")
    p.add_argument("--dpi",         type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 注意力权重提取器（Hook 机制）
# ──────────────────────────────────────────────────────────────

class AttentionExtractor:
    """
    通过 forward hook 从 CLIP Transformer 各层提取多头注意力权重。

    CLIP ViT-L/14 文本编码器共 12 层 ResidualAttentionBlock，
    每层的 attn 是 nn.MultiheadAttention。

    注意：nn.MultiheadAttention 在 need_weights=True 时返回
    平均后的注意力权重，形状为 (tgt_len, src_len)。
    """

    def __init__(self, clip_model):
        self.clip_model  = clip_model
        self._hooks      = []
        self._attn_maps  = {}   # layer_idx -> (seq_len, seq_len)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output = (attn_output, attn_weights)
            # attn_weights shape: (batch, tgt_len, src_len) or (tgt_len, src_len)
            if isinstance(output, tuple) and len(output) >= 2:
                weights = output[1]
                if weights is not None:
                    # 取 batch 维度的均值（通常 batch=1）
                    if weights.dim() == 3:
                        weights = weights.mean(0)
                    self._attn_maps[layer_idx] = weights.detach().cpu()
        return hook

    def register(self):
        """注册所有层的 hook。"""
        for i, block in enumerate(self.clip_model.transformer.resblocks):
            # 替换 forward，让 MultiheadAttention 返回 weights
            h = block.attn.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def remove(self):
        """移除所有 hook，恢复原始状态。"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_attn_maps(self) -> Dict[int, torch.Tensor]:
        return dict(self._attn_maps)

    def clear(self):
        self._attn_maps.clear()


# ──────────────────────────────────────────────────────────────
# 构建两种软提示序列（无/有图注入）
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def build_prompt_sequences(
    model     : RGIPCOL,
    attr      : str,
    obj       : str,
    device    : torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    构建两种软提示序列嵌入。

    Returns
    -------
    seq_clip : (1, seq_len, D)  — 无图注入（CLIP原始词汇嵌入）
    seq_rgcn : (1, seq_len, D)  — 有图注入（RGCN更新后嵌入）
    token_names : list[str]     — 各位置的名称标签
    """
    clip_adapter    = model.clip
    prompt_learner  = model.prompt_learner
    rgcn            = model.rgcn
    k               = prompt_learner.prefix_length

    # ── 1. 从词汇表查找 attr/obj 的全局索引 ──
    if attr not in model.attr2idx:
        raise ValueError(f"attr '{attr}' 不在词汇表中")
    if obj not in model.obj2idx:
        raise ValueError(f"obj '{obj}' 不在词汇表中")

    attr_idx = model.attr2idx[attr]
    obj_idx  = model.obj2idx[obj]

    # ── 2. CLIP 原始词汇嵌入（无图注入） ──
    attr_token = clip_adapter.tokenize([attr])
    obj_token  = clip_adapter.tokenize([obj])
    attr_clip_emb = clip_adapter.get_token_embedding(attr_token)[:, 1, :]  # 跳过 SOS
    obj_clip_emb  = clip_adapter.get_token_embedding(obj_token)[:, 1, :]

    # ── 3. RGCN 更新后嵌入（有图注入） ──
    attr_embs_all, obj_embs_all = rgcn()
    attr_rgcn_emb = attr_embs_all[attr_idx].unsqueeze(0)  # (1, D)
    obj_rgcn_emb  = obj_embs_all[obj_idx].unsqueeze(0)

    # ── 4. 构建完整序列嵌入 ──
    def _build_seq(attr_emb, obj_emb):
        """拼接 [SOS, θ₁..θₖ, attr, obj, EOS]，叠加位置编码"""
        P, D   = 1, attr_emb.size(-1)
        sos    = prompt_learner.sos_emb.unsqueeze(0).unsqueeze(0)   # (1,1,D)
        prefix = prompt_learner.prefix_vectors.unsqueeze(0)          # (1,k,D)
        a_emb  = attr_emb.unsqueeze(0) if attr_emb.dim() == 2 else attr_emb.unsqueeze(0)
        o_emb  = obj_emb.unsqueeze(0)  if obj_emb.dim() == 2  else obj_emb.unsqueeze(0)
        eos    = prompt_learner.eos_emb.unsqueeze(0).unsqueeze(0)    # (1,1,D)

        # attr/obj 嵌入保证维度为 (1,1,D)
        if a_emb.dim() == 3 and a_emb.size(0) == 1 and a_emb.size(1) != 1:
            a_emb = a_emb.unsqueeze(1) if a_emb.dim() == 2 else a_emb
        a_tok = attr_emb.unsqueeze(0).unsqueeze(0) if attr_emb.dim() == 1 else attr_emb.unsqueeze(0)
        o_tok = obj_emb.unsqueeze(0).unsqueeze(0)  if obj_emb.dim() == 1 else obj_emb.unsqueeze(0)

        seq = torch.cat([sos, prefix, a_tok, o_tok, eos], dim=1)  # (1, k+4, D)

        # 叠加位置编码
        seq_len = seq.size(1)
        pos_emb = clip_adapter.positional_embedding[:seq_len, :].unsqueeze(0)
        seq = seq + pos_emb
        return seq.to(device)

    seq_clip = _build_seq(attr_clip_emb.squeeze(0), obj_clip_emb.squeeze(0))
    seq_rgcn = _build_seq(attr_rgcn_emb.squeeze(0), obj_rgcn_emb.squeeze(0))

    # ── 5. Token 名称（用于 x/y 轴标签） ──
    token_names = (
        ["[SOS]"]
        + [f"θ{i+1}" for i in range(k)]
        + [f"[{attr}]", f"[{obj}]", "[EOS]"]
    )

    return seq_clip, seq_rgcn, token_names


# ──────────────────────────────────────────────────────────────
# 前向传播并提取注意力权重
# ──────────────────────────────────────────────────────────────

def forward_and_extract(
    model      : RGIPCOL,
    seq        : torch.Tensor,   # (1, seq_len, D)
    extractor  : AttentionExtractor,
    device     : torch.device,
) -> Dict[int, np.ndarray]:
    """
    将序列送入 CLIP 文本 Transformer，通过 hook 收集注意力权重。

    CLIP 的 MultiheadAttention 默认 need_weights=False，
    需要 monkey-patch 一下才能拿到权重。
    """
    clip_model  = model.clip.clip_model
    seq_len     = seq.size(1)

    # ── Monkey-patch：强制 need_weights=True ──
    original_forwards = {}
    for i, block in enumerate(clip_model.transformer.resblocks):
        orig = block.attn.forward

        def make_patched(orig_fwd):
            def patched(query, key, value,
                        key_padding_mask=None,
                        need_weights=False,
                        attn_mask=None):
                return orig_fwd(
                    query, key, value,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,          # 强制返回权重
                    attn_mask=attn_mask,
                )
            return patched

        original_forwards[i] = orig
        block.attn.forward    = make_patched(orig)

    extractor.clear()

    try:
        # CLIP transformer 期望输入为 (seq_len, batch, D)
        x = seq.squeeze(0).to(clip_model.dtype)  # (seq_len, D)

        # 补全到 context_length=77
        context_length = clip_model.context_length
        if seq_len < context_length:
            pad = torch.zeros(
                context_length - seq_len,
                x.size(1),
                dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, pad], dim=0)   # (77, D)

        # 叠加位置编码（已在 build_prompt_sequences 中加过了，这里不重复加）
        # 直接送入 transformer
        x = x.unsqueeze(1)                    # (77, 1, D)
        x = clip_model.transformer(x)

    finally:
        # 恢复原始 forward
        for i, block in enumerate(clip_model.transformer.resblocks):
            block.attn.forward = original_forwards[i]

    # 截取有效长度的注意力子矩阵
    attn_maps = {}
    for layer_idx, attn in extractor.get_attn_maps().items():
        # attn shape: (77, 77) 或 (1, 77, 77) 等
        if attn.dim() == 3:
            attn = attn.squeeze(0)
        # 只保留 seq_len × seq_len 的有效部分
        attn_valid = attn[:seq_len, :seq_len].float().numpy()
        attn_maps[layer_idx] = attn_valid

    return attn_maps


# ──────────────────────────────────────────────────────────────
# 绘图函数
# ──────────────────────────────────────────────────────────────

def _heatmap(ax, data, xticklabels, yticklabels,
             title="", vmin=0, vmax=None, cmap="Blues",
             annot=False, fontsize=8):
    """通用注意力热力图绘制。"""
    if vmax is None:
        vmax = data.max()
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.3 if annot else 0,
        linecolor="#CCCCCC",
        cbar=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=fontsize, rotation=45)
    ax.tick_params(axis="y", labelsize=fontsize, rotation=0)


def plot_layerwise_heatmaps(
    attn_clip  : Dict[int, np.ndarray],
    attn_rgcn  : Dict[int, np.ndarray],
    token_names: List[str],
    comp_name  : str,
    top_layers : int,
    save_path  : str,
    dpi        : int,
):
    """
    绘制最后 top_layers 层的注意力热力图。
    每层一行，每行3列：无图注入 | 有图注入 | 差异图
    """
    total_layers = max(attn_clip.keys()) + 1
    layer_indices = list(range(
        max(0, total_layers - top_layers), total_layers
    ))

    n_rows = len(layer_indices)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        facecolor="white",
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Without Graph Injection\n(CLIP token embeddings)",
        "With Graph Injection\n(RGCN-updated embeddings)",
        "Difference\n(Graph Injection − No Injection)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=10,
                               color=["#1A5276", "#1E8449", "#922B21"][col])

    for row, layer_idx in enumerate(layer_indices):
        a_clip = attn_clip.get(layer_idx)
        a_rgcn = attn_rgcn.get(layer_idx)
        if a_clip is None or a_rgcn is None:
            continue

        a_diff = a_rgcn - a_clip

        # 无图注入
        _heatmap(
            axes[row, 0], a_clip,
            xticklabels=token_names,
            yticklabels=token_names,
            title=f"Layer {layer_idx + 1}",
            cmap="Blues",
            fontsize=7,
        )

        # 有图注入
        _heatmap(
            axes[row, 1], a_rgcn,
            xticklabels=token_names,
            yticklabels=token_names,
            title=f"Layer {layer_idx + 1}",
            cmap="Greens",
            fontsize=7,
        )

        # 差异图
        max_abs = max(abs(a_diff.min()), abs(a_diff.max())) + 1e-6
        _heatmap(
            axes[row, 2], a_diff,
            xticklabels=token_names,
            yticklabels=token_names,
            title=f"Layer {layer_idx + 1}",
            vmin=-max_abs, vmax=max_abs,
            cmap="RdBu_r",
            fontsize=7,
        )

        # 左侧行标注
        axes[row, 0].set_ylabel(
            f"Layer {layer_idx + 1}", fontsize=9, fontweight="bold"
        )

    fig.suptitle(
        f"Attention Weight Visualization — '{comp_name}'\n"
        f"Soft Prompt Sequence: {' → '.join(token_names)}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 逐层热力图已保存：{save_path}")


def plot_eos_attention_barplot(
    attn_clip  : Dict[int, np.ndarray],
    attn_rgcn  : Dict[int, np.ndarray],
    token_names: List[str],
    comp_name  : str,
    top_layers : int,
    save_path  : str,
    dpi        : int,
):
    """
    绘制 EOS 位置对各 token 的注意力均值对比柱状图。

    EOS 位置是最终输出向量的来源，其注意力分布直接反映
    模型在编码组合概念时"关注"了序列的哪些部分。
    """
    total_layers = max(attn_clip.keys()) + 1
    layer_indices = list(range(
        max(0, total_layers - top_layers), total_layers
    ))

    # 对 top_layers 层取均值
    def _avg_eos_attn(attn_dict):
        eos_idx = len(token_names) - 1   # EOS 是最后一个 token
        attn_list = []
        for li in layer_indices:
            a = attn_dict.get(li)
            if a is not None and eos_idx < a.shape[0]:
                attn_list.append(a[eos_idx, :len(token_names)])
        if not attn_list:
            return np.zeros(len(token_names))
        return np.mean(attn_list, axis=0)

    eos_clip = _avg_eos_attn(attn_clip)
    eos_rgcn = _avg_eos_attn(attn_rgcn)

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    x      = np.arange(len(token_names))
    width  = 0.35

    # 左图：绝对值对比
    ax = axes[0]
    bars1 = ax.bar(x - width/2, eos_clip, width, label="No Graph Injection",
                   color="#2E86C1", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, eos_rgcn, width, label="With Graph Injection",
                   color="#1E8449", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(token_names, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Average Attention Weight", fontsize=10)
    ax.set_title(
        f"EOS Token Attention Distribution\n(avg. over last {top_layers} layers)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(eos_clip.max(), eos_rgcn.max()) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 高亮 attr / obj 位置
    k = len(token_names) - 4   # prefix length
    attr_pos = k + 1
    obj_pos  = k + 2
    for pos in [attr_pos, obj_pos]:
        ax.axvspan(pos - 0.5, pos + 0.5, color="yellow", alpha=0.15, zorder=0)

    # 右图：差异柱状图（有图注入 − 无图注入）
    ax2   = axes[1]
    diff  = eos_rgcn - eos_clip
    colors = ["#C0392B" if d < 0 else "#1E8449" for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(token_names, fontsize=10, rotation=30, ha="right")
    ax2.set_ylabel("Δ Attention Weight (Graph − No Graph)", fontsize=10)
    ax2.set_title(
        "Attention Shift caused by Graph Injection\n(positive = increased attention)",
        fontsize=11, fontweight="bold",
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for pos in [attr_pos, obj_pos]:
        ax2.axvspan(pos - 0.5, pos + 0.5, color="yellow", alpha=0.15, zorder=0)

    # 注释黄色高亮含义
    from matplotlib.patches import Patch
    legend_elem = Patch(facecolor="yellow", alpha=0.3, label="attr / obj positions")
    for ax_ in axes:
        handles, labels = ax_.get_legend_handles_labels()
        ax_.legend(handles=handles + [legend_elem],
                   labels=labels + ["attr / obj positions"],
                   fontsize=8, loc="upper right")

    fig.suptitle(
        f"EOS Attention Analysis — '{comp_name}'",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ EOS 注意力柱状图已保存：{save_path}")


def plot_multi_comp_comparison(
    results    : List[dict],
    top_layers : int,
    save_path  : str,
    dpi        : int,
):
    """
    多组合对比图：每个组合一行，展示有/无图注入的 EOS 注意力差异。
    适合论文中直接作为对比图使用。
    """
    n_comps = len(results)
    if n_comps == 0:
        return

    fig, axes = plt.subplots(
        n_comps, 1,
        figsize=(12, 3.5 * n_comps),
        facecolor="white",
    )
    if n_comps == 1:
        axes = [axes]

    for row, res in enumerate(results):
        ax          = axes[row]
        token_names = res["token_names"]
        eos_clip    = res["eos_clip"]
        eos_rgcn    = res["eos_rgcn"]
        comp_name   = res["comp_name"]

        x     = np.arange(len(token_names))
        width = 0.38

        ax.bar(x - width/2, eos_clip, width,
               label="No Graph Injection",
               color="#5DADE2", alpha=0.9, edgecolor="white")
        ax.bar(x + width/2, eos_rgcn, width,
               label="With Graph Injection",
               color="#52BE80", alpha=0.9, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(token_names, fontsize=9.5)
        ax.set_ylabel("Attn. Weight", fontsize=9)
        ax.set_title(f"'{comp_name}'", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(0, max(eos_clip.max(), eos_rgcn.max()) * 1.3)

        # 高亮 attr / obj 位置
        k = len(token_names) - 4
        for pos in [k + 1, k + 2]:
            ax.axvspan(pos - 0.5, pos + 0.5, color="#F9E79F", alpha=0.5, zorder=0)

    fig.suptitle(
        f"EOS Token Attention: Graph Injection Effect\n"
        f"(avg. over last {top_layers} transformer layers)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 多组合对比图已保存：{save_path}")


def plot_diff_summary_heatmap(
    results    : List[dict],
    top_layers : int,
    save_path  : str,
    dpi        : int,
):
    """
    所有组合的注意力差异汇总热力图。
    行 = 组合名称，列 = token 位置，值 = Δ attention (rgcn − clip)
    直观展示图注入对不同组合的影响是否一致。
    """
    if not results:
        return

    token_names = results[0]["token_names"]
    diff_matrix = np.array([
        res["eos_rgcn"] - res["eos_clip"] for res in results
    ])   # (n_comps, seq_len)

    comp_names = [res["comp_name"] for res in results]

    fig, ax = plt.subplots(
        figsize=(max(8, len(token_names) * 0.9), max(4, len(results) * 0.7)),
        facecolor="white",
    )
    max_abs = max(abs(diff_matrix.min()), abs(diff_matrix.max())) + 1e-6
    sns.heatmap(
        diff_matrix,
        ax=ax,
        xticklabels=token_names,
        yticklabels=comp_names,
        vmin=-max_abs, vmax=max_abs,
        cmap="RdBu_r",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Δ Attention (RGCN − CLIP)", "shrink": 0.8},
    )
    ax.set_title(
        f"Attention Shift Summary (EOS position, avg. last {top_layers} layers)\n"
        "Red = Graph injection reduces attention | Blue = increases attention",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 差异汇总热力图已保存：{save_path}")


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. 加载配置与数据集 ──
    print("[1/4] 加载配置与数据集...")
    cfg = load_config(args.config)
    cfg["num_workers"] = 0
    _, train_dataset = get_dataloader(cfg, "train")
    dataset_name = cfg.get("dataset", "dataset")
    print(f"      数据集：{dataset_name}")

    # ── 2. 加载模型 ──
    print("[2/4] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RGIPCOL(cfg=cfg, dataset=train_dataset, device=device).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    clip_model = model.clip.clip_model
    total_layers = len(clip_model.transformer.resblocks)
    print(f"      CLIP 文本 Transformer 共 {total_layers} 层")

    # ── 3. 注册注意力 Hook ──
    extractor = AttentionExtractor(clip_model)
    extractor.register()
    print(f"      注意力 Hook 已注册")

    # ── 4. 逐组合提取 & 绘图 ──
    print(f"\n[3/4] 处理 {len(args.comps)} 个组合...")
    all_results = []

    for comp_str in args.comps:
        parts = comp_str.strip().split()
        if len(parts) < 2:
            print(f"  ⚠ 跳过格式错误的组合：'{comp_str}'（需要 'attr obj'）")
            continue
        # 支持多词属性/物体（取首词为attr，其余为obj）
        attr = parts[0]
        obj  = " ".join(parts[1:])

        print(f"\n  处理组合：'{attr} {obj}'")

        try:
            # 构建两种序列
            seq_clip, seq_rgcn, token_names = build_prompt_sequences(
                model, attr, obj, device
            )
        except ValueError as e:
            print(f"  ⚠ {e}，跳过")
            continue

        # 提取无图注入的注意力权重
        extractor.clear()
        with torch.no_grad():
            attn_clip = forward_and_extract(model, seq_clip, extractor, device)

        # 提取有图注入的注意力权重
        extractor.clear()
        with torch.no_grad():
            attn_rgcn = forward_and_extract(model, seq_rgcn, extractor, device)

        # 计算 EOS 平均注意力
        total_layers_cnt = max(attn_clip.keys()) + 1
        layer_indices    = list(range(
            max(0, total_layers_cnt - args.top_layers), total_layers_cnt
        ))
        eos_idx = len(token_names) - 1

        def _avg_eos(attn_dict):
            lst = [attn_dict[li][eos_idx, :len(token_names)]
                   for li in layer_indices if li in attn_dict
                   and eos_idx < attn_dict[li].shape[0]]
            return np.mean(lst, axis=0) if lst else np.zeros(len(token_names))

        eos_clip_arr = _avg_eos(attn_clip)
        eos_rgcn_arr = _avg_eos(attn_rgcn)

        all_results.append({
            "comp_name"  : f"{attr} {obj}",
            "token_names": token_names,
            "eos_clip"   : eos_clip_arr,
            "eos_rgcn"   : eos_rgcn_arr,
            "attn_clip"  : attn_clip,
            "attn_rgcn"  : attn_rgcn,
        })

        # 每个组合单独出图
        safe_name = f"{attr}_{obj}".replace(" ", "_")

        # ① 逐层热力图
        plot_layerwise_heatmaps(
            attn_clip, attn_rgcn, token_names,
            comp_name  = f"{attr} {obj}",
            top_layers = args.top_layers,
            save_path  = os.path.join(
                args.save_dir, f"layerwise_{safe_name}.png"
            ),
            dpi=args.dpi,
        )

        # ② EOS 注意力柱状图
        plot_eos_attention_barplot(
            attn_clip, attn_rgcn, token_names,
            comp_name  = f"{attr} {obj}",
            top_layers = args.top_layers,
            save_path  = os.path.join(
                args.save_dir, f"eos_barplot_{safe_name}.png"
            ),
            dpi=args.dpi,
        )

    # ── 5. 多组合汇总图 ──
    print("\n[4/4] 生成汇总图...")

    if all_results:
        plot_multi_comp_comparison(
            all_results,
            top_layers = args.top_layers,
            save_path  = os.path.join(args.save_dir, "multi_comp_comparison.png"),
            dpi=args.dpi,
        )
        plot_diff_summary_heatmap(
            all_results,
            top_layers = args.top_layers,
            save_path  = os.path.join(args.save_dir, "diff_summary_heatmap.png"),
            dpi=args.dpi,
        )

    # 移除 hook
    extractor.remove()

    print(f"\n✓ 全部完成！输出目录：{os.path.abspath(args.save_dir)}")
    print("  生成文件：")
    for f in sorted(os.listdir(args.save_dir)):
        print(f"    {f}")


if __name__ == "__main__":
    main()