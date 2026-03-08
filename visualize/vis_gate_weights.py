"""
visualize/vis_gate_weights.py
------------------------------
GateFusion 门控权重 α 分布可视化

分析训练后模型中 GateFusion 的门控值 α（对每个节点、每个特征维度），
回答核心问题：模型在不同节点类型（attr/obj/comp）上
更依赖原始 CLIP 特征（α→1）还是 LLM 描述特征（α→0）？

生成四张子图：
  1. 整体 α 分布直方图（所有节点、所有维度）
  2. 按节点类型（attr/obj/comp）分组的 α 分布小提琴图
  3. 每个特征维度的平均 α（768 维度分布条形图）
  4. 节点级 α 均值的散点图（按节点索引，attr/obj/comp 不同颜色）

用法：
    python visualize/vis_gate_weights.py \
        --config     configs/mit_states_config.yaml \
        --checkpoint saved_models/mit-states/xxx/best_model.pt \
        --output     visualize/output/gate_weights_mit.pdf
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.data_utils import load_config
from data.dataset import get_dataloader
from models.rgipcol import RGIPCOL
from utils.checkpoint import load_checkpoint


# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=str, required=True)
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--output",      type=str,
                   default="visualize/output/gate_weights.pdf")
    return p.parse_args()


# ──────────────────────────────────────────────
# 提取 α 值
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_alpha(model) -> np.ndarray:
    """
    计算所有节点的门控值 α：
        α = sigmoid(gate_linear( concat(node_emb_orig, node_emb_llm) ))
    返回 (N, D) 的 numpy 数组。
    """
    rgcn = model.rgcn
    h_orig = rgcn.node_emb_orig   # (N, D)
    h_llm  = rgcn.node_emb_llm    # (N, D)

    concat = torch.cat([h_orig, h_llm], dim=-1)   # (N, 2D)
    alpha  = torch.sigmoid(rgcn.gate_fusion.gate_linear(concat))  # (N, D)

    return alpha.cpu().numpy()


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────

TYPE_COLORS = {
    "attr": "#4C9BE8",
    "obj" : "#F28C28",
    "comp": "#52C26A",
}


def draw_all(alpha: np.ndarray, num_attrs: int, num_objs: int,
             num_comps: int, output_path: str, dataset_name: str):
    """生成四个子图"""
    N, D = alpha.shape

    # ── 节点类型切片 ──
    attr_alpha = alpha[:num_attrs]
    obj_alpha  = alpha[num_attrs : num_attrs + num_objs]
    comp_alpha = alpha[num_attrs + num_objs :]

    # ── 每节点的平均 α ──
    node_mean = alpha.mean(axis=1)  # (N,)
    attr_node_mean = node_mean[:num_attrs]
    obj_node_mean  = node_mean[num_attrs : num_attrs + num_objs]
    comp_node_mean = node_mean[num_attrs + num_objs :]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"GateFusion α Distribution Analysis\n"
        f"(α→1: prefer CLIP orig, α→0: prefer LLM desc) | dataset: {dataset_name}",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ────────────────────────────────────────
    # 子图1：整体 α 分布直方图
    # ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(alpha.ravel(), bins=60, color="#7B68EE",
             alpha=0.75, edgecolor="white", linewidth=0.3)
    ax1.axvline(alpha.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"mean={alpha.mean():.3f}")
    ax1.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                label="α=0.5 (balanced)")
    ax1.set_xlabel("Gate Weight α", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Overall α Distribution\n(all nodes × all dims)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # 标注解读区域
    ax1.axvspan(0.0, 0.4, alpha=0.06, color="green",
                label="LLM-dominant (α<0.4)")
    ax1.axvspan(0.6, 1.0, alpha=0.06, color="blue",
                label="CLIP-dominant (α>0.6)")
    ax1.text(0.1, ax1.get_ylim()[1] * 0.85, "LLM\ndominant",
             fontsize=8, color="green", ha="center")
    ax1.text(0.9, ax1.get_ylim()[1] * 0.85, "CLIP\ndominant",
             fontsize=8, color="blue", ha="center")

    # ────────────────────────────────────────
    # 子图2：按节点类型的小提琴图
    # ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    # 使用每个节点的 α 均值，而非所有维度展开（避免太多数据点遮挡）
    data_groups = [
        ("attr", attr_node_mean),
        ("obj",  obj_node_mean),
        ("comp", comp_node_mean),
    ]

    parts = ax2.violinplot(
        [g[1] for g in data_groups],
        positions=[1, 2, 3],
        showmedians=True,
        showextrema=True,
    )
    # 着色
    type_list = ["attr", "obj", "comp"]
    for i, (pc_key, pc) in enumerate(parts.items()):
        if pc_key == "bodies":
            for j, body in enumerate(pc):
                body.set_facecolor(TYPE_COLORS[type_list[j]])
                body.set_alpha(0.65)
        else:
            pc.set_edgecolor("black")
            pc.set_linewidth(1.0)

    # 叠加 jitter 散点
    for i, (tname, vals) in enumerate(data_groups):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax2.scatter(np.full(len(vals), i + 1) + jitter, vals,
                    color=TYPE_COLORS[tname], s=15, alpha=0.45, zorder=3)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Attr\n({})".format(num_attrs),
                          "Obj\n({})".format(num_objs),
                          "Comp\n({})".format(num_comps)], fontsize=10)
    ax2.set_ylabel("Node-level Mean α", fontsize=10)
    ax2.set_title("α Distribution by Node Type\n(each point = one node's avg α)",
                  fontsize=11)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.spines[["top", "right"]].set_visible(False)

    # 打印中位数
    for i, (tname, vals) in enumerate(data_groups):
        ax2.text(i + 1, np.median(vals) + 0.03, f"med={np.median(vals):.3f}",
                 ha="center", fontsize=8, color=TYPE_COLORS[tname],
                 fontweight="bold")

    # ────────────────────────────────────────
    # 子图3：每维度平均 α（768 维度条形图）
    # ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    dim_means = alpha.mean(axis=0)   # (D,) 所有节点在每个维度的平均 α

    # 按节点类型分别画线图，更清晰
    ax3.fill_between(range(D), alpha[:num_attrs].mean(axis=0),
                     alpha=0.4, color=TYPE_COLORS["attr"], label="attr avg")
    ax3.fill_between(range(D),
                     alpha[num_attrs:num_attrs+num_objs].mean(axis=0),
                     alpha=0.4, color=TYPE_COLORS["obj"],  label="obj avg")
    ax3.fill_between(range(D),
                     alpha[num_attrs+num_objs:].mean(axis=0),
                     alpha=0.3, color=TYPE_COLORS["comp"], label="comp avg")
    ax3.plot(dim_means, color="black", linewidth=0.8, alpha=0.6, label="all avg")
    ax3.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)

    ax3.set_xlabel("Feature Dimension", fontsize=10)
    ax3.set_ylabel("Mean α", fontsize=10)
    ax3.set_title("Per-Dimension Mean α\n(which dims favor CLIP vs LLM)",
                  fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.spines[["top", "right"]].set_visible(False)

    # 标注高 LLM 维度（α 最低的 top 10 维度）
    llm_dims = np.argsort(dim_means)[:10]
    ax3.scatter(llm_dims, dim_means[llm_dims],
                color="green", s=30, zorder=5, label="LLM-dominant dims")

    # ────────────────────────────────────────
    # 子图4：节点级 α 均值散点图
    # ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    x_attr = np.arange(num_attrs)
    x_obj  = np.arange(num_attrs, num_attrs + num_objs)
    x_comp = np.arange(num_attrs + num_objs, N)

    ax4.scatter(x_attr, attr_node_mean, c=TYPE_COLORS["attr"],
                s=40, alpha=0.8, zorder=3, label=f"attr ({num_attrs})")
    ax4.scatter(x_obj,  obj_node_mean,  c=TYPE_COLORS["obj"],
                s=40, alpha=0.8, zorder=3, label=f"obj ({num_objs})")
    ax4.scatter(x_comp, comp_node_mean, c=TYPE_COLORS["comp"],
                s=8,  alpha=0.4, zorder=2, label=f"comp ({num_comps})")

    ax4.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax4.axhline(alpha.mean(), color="red", linestyle="--",
                linewidth=1.0, alpha=0.7, label=f"global mean={alpha.mean():.3f}")

    ax4.set_xlabel("Node Index", fontsize=10)
    ax4.set_ylabel("Node-level Mean α", fontsize=10)
    ax4.set_title("Per-Node Mean α\n(attr/obj/comp colored)",
                  fontsize=11)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=8, loc="lower right")
    ax4.spines[["top", "right"]].set_visible(False)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[Gate] 已保存：{output_path}")

    # ── 打印关键统计 ──
    print(f"\n[Gate] 门控权重统计：")
    print(f"  整体均值 α = {alpha.mean():.4f}  (>0.5 偏向 CLIP 原始特征)")
    for tname, vals in data_groups:
        print(f"  {tname:4s} 节点均值 α = {vals.mean():.4f}  "
              f"[{vals.min():.3f}, {vals.max():.3f}]")

    clip_dom = (alpha.mean(axis=1) > 0.5).mean() * 100
    llm_dom  = (alpha.mean(axis=1) < 0.5).mean() * 100
    print(f"\n  CLIP 主导节点（avg α>0.5）：{clip_dom:.1f}%")
    print(f"  LLM  主导节点（avg α<0.5）：{llm_dom:.1f}%")

    llm_dom_dims = (dim_means < 0.5).sum()
    print(f"  LLM 主导特征维度（avg α<0.5）：{llm_dom_dims}/{D} = {llm_dom_dims/D*100:.1f}%")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_dataset = get_dataloader(cfg, "train")

    model = RGIPCOL(cfg=cfg, dataset=train_dataset, device=device).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    print("[Gate] 提取门控权重 α ...")
    alpha = extract_alpha(model)
    print(f"[Gate] α 矩阵大小：{alpha.shape}  (N nodes × D dims)")

    rgcn = model.rgcn
    draw_all(
        alpha,
        num_attrs    = rgcn.num_attrs,
        num_objs     = rgcn.num_objs,
        num_comps    = rgcn.num_comps,
        output_path  = args.output,
        dataset_name = cfg.get("dataset", ""),
    )


if __name__ == "__main__":
    main()