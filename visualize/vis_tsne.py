"""
visualize/vis_tsne.py
---------------------
节点嵌入 t-SNE 可视化：RGCN 传播前后对比

生成两张并排子图：
  左图：传播前（gate_fusion 输出，即初始节点表示）
  右图：传播后（RGCN 最终输出）

节点按类型着色：attr(蓝) / obj(橙) / comp(绿)
同时可选：对 comp 节点按所属 attr 或 obj 进一步着色，
          观察相同属性/物体的组合是否聚类

用法：
    python visualize/vis_tsne.py \
        --config  configs/mit_states_config.yaml \
        --checkpoint saved_models/mit-states/xxx/best_model.pt \
        --output  visualize/output/tsne_mit.pdf \
        --color_by node_type          # or "attr" or "obj"
        --max_comp 300                # comp 节点采样数（太多会拥挤）
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
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

import yaml
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
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="模型检查点（不提供则使用随机初始化权重）")
    p.add_argument("--output",      type=str,
                   default="visualize/output/tsne.pdf")
    p.add_argument("--color_by",    type=str, default="node_type",
                   choices=["node_type", "attr", "obj"],
                   help="节点着色依据")
    p.add_argument("--max_comp",    type=int, default=300,
                   help="最多展示的 comp 节点数（随机采样）")
    p.add_argument("--perplexity",  type=float, default=30.0)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────
# 特征提取：拿到传播前后的节点嵌入
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, device):
    """
    提取两组节点嵌入：
      before : gate_fusion(node_emb_orig, node_emb_llm)  ← 传播前初始表示
      after  : RGCN 传播后的完整节点表示（包含 attr+obj+comp）

    Returns
    -------
    before_emb : (N, D) numpy
    after_emb  : (N, D) numpy
    """
    model.eval()
    rgcn = model.rgcn

    # ── 传播前：门控融合后的初始节点嵌入 ──
    before_fused = rgcn.gate_fusion(
        rgcn.node_emb_orig,
        rgcn.node_emb_llm
    )  # (N, D)
    before_proj = rgcn.input_proj(before_fused)  # (N, hidden_dim)

    # ── 传播后：完整 RGCN forward，但需要获取所有节点（而不仅是 attr/obj） ──
    # 复用 forward 内部逻辑，临时 hook 最后一层输出
    x = before_proj.clone()
    for layer, norm in zip(rgcn.rgcn_layers, rgcn.layer_norms):
        x_new = layer(x, rgcn.edge_index, rgcn.edge_type, rgcn.edge_weight)
        x_new = norm(x_new)
        if x.size(-1) == x_new.size(-1):
            import torch.nn.functional as F
            x = F.relu(x_new + x)
        else:
            x = F.relu(x_new)
    after_all = x  # (N, output_dim)

    return (
        before_proj.cpu().numpy(),
        after_all.cpu().numpy(),
    )


# ──────────────────────────────────────────────
# 构建节点元信息（用于着色与标签）
# ──────────────────────────────────────────────

def build_node_meta(model, dataset, max_comp: int, seed: int):
    """
    返回 node_types, node_attr_ids, node_obj_ids（与节点顺序对齐）
    以及各段的 slice 索引。
    """
    rgcn        = model.rgcn
    num_attrs   = rgcn.num_attrs
    num_objs    = rgcn.num_objs
    num_comps   = rgcn.num_comps
    train_pairs = dataset.train_pairs

    # ── comp 节点采样 ──
    rng = np.random.RandomState(seed)
    if num_comps > max_comp:
        comp_sel = sorted(rng.choice(num_comps, max_comp, replace=False))
    else:
        comp_sel = list(range(num_comps))

    # ── 为每个节点记录类型 ──
    types    = (["attr"] * num_attrs +
                ["obj"]  * num_objs  +
                ["comp"] * num_comps)   # 全量，后面再 mask

    # ── attr_id / obj_id（comp 节点归属）──
    attr_ids = [-1] * (num_attrs + num_objs)
    obj_ids  = [-1] * (num_attrs + num_objs)
    for i, (a, o) in enumerate(train_pairs):
        attr_ids.append(dataset.attr2idx.get(a, -1))
        obj_ids.append(dataset.obj2idx.get(o, -1))

    # ── 只保留选出的 comp 索引 ──
    keep_indices = (
        list(range(num_attrs)) +
        list(range(num_attrs, num_attrs + num_objs)) +
        [num_attrs + num_objs + i for i in comp_sel]
    )

    return {
        "keep_indices": keep_indices,
        "types"       : [types[i] for i in keep_indices],
        "attr_ids"    : [attr_ids[i] for i in keep_indices],
        "obj_ids"     : [obj_ids[i] for i in keep_indices],
        "num_attrs"   : num_attrs,
        "num_objs"    : num_objs,
        "comp_sel"    : comp_sel,
        "attrs"       : dataset.attrs,
        "objs"        : dataset.objs,
        "train_pairs" : train_pairs,
    }


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────

TYPE_COLORS = {
    "attr": "#4C9BE8",   # 蓝
    "obj" : "#F28C28",   # 橙
    "comp": "#52C26A",   # 绿
}
TYPE_SIZES  = {"attr": 60, "obj": 60, "comp": 20}
TYPE_ALPHA  = {"attr": 0.9, "obj": 0.9, "comp": 0.5}
TYPE_ZORDER = {"attr": 3, "obj": 3, "comp": 1}


def run_tsne(emb: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components = 2,
        perplexity   = min(perplexity, max(5, emb.shape[0] // 3)),
        random_state = seed,
        n_iter       = 1000,
        init         = "pca",
        learning_rate= "auto",
    )
    return tsne.fit_transform(emb)


def draw_panel(
    ax,
    coords  : np.ndarray,
    meta    : dict,
    color_by: str,
    title   : str,
):
    """在单个 Axes 上绘制 t-SNE 散点图"""
    types    = meta["types"]
    attr_ids = meta["attr_ids"]
    obj_ids  = meta["obj_ids"]

    if color_by == "node_type":
        colors = [TYPE_COLORS[t] for t in types]
        sizes  = [TYPE_SIZES[t]  for t in types]
        alphas = [TYPE_ALPHA[t]  for t in types]
        zorder = [TYPE_ZORDER[t] for t in types]

        for t in ["comp", "obj", "attr"]:   # comp 先画（在底层）
            mask = [i for i, tt in enumerate(types) if tt == t]
            if not mask:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c      = TYPE_COLORS[t],
                s      = TYPE_SIZES[t],
                alpha  = TYPE_ALPHA[t],
                zorder = TYPE_ZORDER[t],
                linewidths = 0,
                label  = t,
            )
        # 图例
        patches = [mpatches.Patch(color=TYPE_COLORS[t], label=t)
                   for t in ["attr", "obj", "comp"]]
        ax.legend(handles=patches, fontsize=9, loc="best",
                  framealpha=0.7)

    elif color_by == "attr":
        # comp 节点按 attr 着色，attr/obj 节点灰色
        cmap = plt.cm.get_cmap("tab20", len(meta["attrs"]))
        for i, (x, y) in enumerate(coords):
            t = types[i]
            if t == "comp":
                aid = attr_ids[i]
                c   = cmap(aid) if aid >= 0 else "gray"
                ax.scatter(x, y, c=[c], s=15, alpha=0.5, linewidths=0)
            elif t == "attr":
                ax.scatter(x, y, c="black", s=80, marker="^",
                           alpha=0.9, zorder=5, linewidths=0.5,
                           edgecolors="white")
            else:
                ax.scatter(x, y, c="lightgray", s=40, alpha=0.4,
                           linewidths=0, zorder=1)

    elif color_by == "obj":
        cmap = plt.cm.get_cmap("tab20", len(meta["objs"]))
        for i, (x, y) in enumerate(coords):
            t = types[i]
            if t == "comp":
                oid = obj_ids[i]
                c   = cmap(oid) if oid >= 0 else "gray"
                ax.scatter(x, y, c=[c], s=15, alpha=0.5, linewidths=0)
            elif t == "obj":
                ax.scatter(x, y, c="black", s=80, marker="s",
                           alpha=0.9, zorder=5, linewidths=0.5,
                           edgecolors="white")
            else:
                ax.scatter(x, y, c="lightgray", s=40, alpha=0.4,
                           linewidths=0, zorder=1)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","left","bottom"]].set_visible(False)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── 加载配置与数据 ──
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_dataset = get_dataloader(cfg, "train")

    # ── 构建模型 ──
    model = RGIPCOL(cfg=cfg, dataset=train_dataset, device=device).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    print("[t-SNE] 提取节点嵌入...")
    before_emb, after_emb = extract_embeddings(model, device)

    meta = build_node_meta(model, train_dataset, args.max_comp, args.seed)
    ki   = meta["keep_indices"]

    before_sel = before_emb[ki]
    after_sel  = after_emb[ki]

    print(f"[t-SNE] 节点数：attr={meta['num_attrs']}, "
          f"obj={meta['num_objs']}, comp={len(meta['comp_sel'])}")
    print(f"[t-SNE] 运行 t-SNE（可能需要1-2分钟）...")

    coords_before = run_tsne(before_sel, args.perplexity, args.seed)
    coords_after  = run_tsne(after_sel,  args.perplexity, args.seed)

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Node Embedding t-SNE — {cfg.get('dataset','')}\n"
        f"(colored by {args.color_by})",
        fontsize=14, y=1.01
    )

    draw_panel(axes[0], coords_before, meta, args.color_by,
               "Before RGCN Propagation\n(Gate-Fused Initial)")
    draw_panel(axes[1], coords_after,  meta, args.color_by,
               "After RGCN Propagation\n(Graph-Updated)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"[t-SNE] 已保存：{args.output}")

    # ── 额外输出：per-type 的 intra-class 紧密度（可选量化指标） ──
    _print_cluster_stats(coords_before, coords_after, meta)


def _print_cluster_stats(before, after, meta):
    """量化 RGCN 前后 comp 节点聚类紧密度变化（同 attr 的组合是否更近）"""
    types    = np.array(meta["types"])
    attr_ids = np.array(meta["attr_ids"])

    comp_mask = (types == "comp")
    if comp_mask.sum() < 10:
        return

    def mean_intra_dist(coords, labels):
        """同标签节点之间的平均 L2 距离（越小越聚集）"""
        unique = np.unique(labels[labels >= 0])
        dists  = []
        for lb in unique:
            sel = coords[labels == lb]
            if len(sel) < 2:
                continue
            # 只取随机 50 对，避免 O(n^2)
            idx = np.random.choice(len(sel), min(len(sel), 10), replace=False)
            sel = sel[idx]
            d   = np.mean(np.linalg.norm(sel[:, None] - sel[None, :], axis=-1))
            dists.append(d)
        return np.mean(dists) if dists else float("nan")

    comp_before = before[comp_mask]
    comp_after  = after[comp_mask]
    comp_attr   = attr_ids[comp_mask]

    dist_b = mean_intra_dist(comp_before, comp_attr)
    dist_a = mean_intra_dist(comp_after,  comp_attr)

    print(f"\n[Cluster Stats] Comp 节点按 attr 分组的平均组内 L2 距离：")
    print(f"  传播前：{dist_b:.4f}")
    print(f"  传播后：{dist_a:.4f}")
    delta = (dist_b - dist_a) / dist_b * 100 if dist_b > 0 else 0
    print(f"  聚集度提升：{delta:.1f}%（正值表示传播后同属性组合更紧密）")


if __name__ == "__main__":
    main()