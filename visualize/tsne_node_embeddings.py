"""
visualize/tsne_node_embeddings.py
----------------------------------
节点嵌入 t-SNE / UMAP 可视化脚本（论文图3-X：图传播前后节点分布对比）

功能：
  1. 加载训练好的 RGIPCOL 模型
  2. 提取图传播「前」（GateFusion输出）和「后」（RGCN输出）的节点嵌入
  3. 用 t-SNE / UMAP 降维到 2D
  4. 绘制四张子图：
       [左上] 传播前  - 按节点类型着色（attr / obj / comp）
       [右上] 传播后  - 按节点类型着色
       [左下] 传播前  - 按「共享属性」关系着色（高亮共享同一属性的节点簇）
       [右下] 传播后  - 按「共享属性」关系着色

使用方法：
    python visualize/tsne_node_embeddings.py \
        --config   configs/mit_states_config.yaml \
        --checkpoint saved_models/mit-states/xxx/best_model.pt \
        --method   tsne \
        --save_dir ./visualize/outputs

依赖：
    pip install scikit-learn matplotlib seaborn
    pip install umap-learn   # 可选，使用 --method umap 时需要
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
import matplotlib

# ── 将项目根目录加入 sys.path ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils import load_config
from data.dataset import get_dataloader
from models.rgipcol import RGIPCOL
from utils.checkpoint import load_checkpoint


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node Embedding t-SNE/UMAP Visualization")
    p.add_argument("--config",      type=str, required=True,
                   help="yaml 配置文件路径")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="模型权重路径（best_model.pt）")
    p.add_argument("--method",      type=str, default="tsne",
                   choices=["tsne", "umap", "both"],
                   help="降维方法（tsne / umap / both）")
    p.add_argument("--perplexity",  type=float, default=30.0,
                   help="t-SNE perplexity 参数")
    p.add_argument("--n_neighbors", type=int,   default=15,
                   help="UMAP n_neighbors 参数")
    p.add_argument("--max_nodes",   type=int,   default=500,
                   help="每类节点最多可视化数量（避免过密）")
    p.add_argument("--save_dir",    type=str,   default="./visualize/outputs",
                   help="图像保存目录")
    p.add_argument("--dpi",         type=int,   default=200,
                   help="输出图像 DPI")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 从 RGCNEncoder 提取传播前后的节点嵌入
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model: RGIPCOL):
    """
    提取两组节点嵌入：
      - before: GateFusion 输出（图传播前）
      - after : RGCN 多层传播后输出

    Returns
    -------
    before : np.ndarray, shape (N, D)
    after  : np.ndarray, shape (N, D)
    meta   : dict，包含 num_attrs / num_objs / num_comps / attrs / objs / train_pairs
    """
    model.eval()
    rgcn = model.rgcn

    # ── 1. GateFusion 输出（传播前） ──
    clip_a, clip_o, clip_c = torch.split(
        rgcn.h_clip, [rgcn.num_attrs, rgcn.num_objs, rgcn.num_comps]
    )
    llm_a, llm_o, llm_c = torch.split(
        rgcn.h_llm, [rgcn.num_attrs, rgcn.num_objs, rgcn.num_comps]
    )

    g_a = torch.sigmoid(rgcn.gate_attr(torch.cat([clip_a, llm_a], dim=-1)))
    h_a = g_a * clip_a + (1.0 - g_a) * llm_a

    g_o = torch.sigmoid(rgcn.gate_obj(torch.cat([clip_o, llm_o], dim=-1)))
    h_o = g_o * clip_o + (1.0 - g_o) * llm_o

    g_c = torch.sigmoid(rgcn.gate_comp(torch.cat([clip_c, llm_c], dim=-1)))
    h_c = g_c * clip_c + (1.0 - g_c) * llm_c

    x_before = torch.cat([h_a, h_o, h_c], dim=0)   # (N, D)
    x_before = rgcn.input_proj(x_before)

    # ── 2. RGCN 多层传播后（传播后） ──
    x = x_before.clone()
    for i, (layer, norm) in enumerate(zip(rgcn.rgcn_layers, rgcn.layer_norms)):
        x_new = layer(x, rgcn.edge_index, rgcn.edge_type)
        x_new = norm(x_new)
        if x.size(-1) == x_new.size(-1):
            x = F.relu(x_new + x)
        else:
            x = F.relu(x_new)

    x_after = x   # (N, D)

    # ── L2 归一化（对齐推理时的空间） ──
    x_before = F.normalize(x_before, dim=-1)
    x_after  = F.normalize(x_after,  dim=-1)

    meta = {
        "num_attrs"   : rgcn.num_attrs,
        "num_objs"    : rgcn.num_objs,
        "num_comps"   : rgcn.num_comps,
        "attrs"       : rgcn.graph_data["attrs"],
        "objs"        : rgcn.graph_data["objs"],
        "train_pairs" : rgcn.graph_data["train_pairs"],
    }

    return (
        x_before.cpu().float().numpy(),
        x_after.cpu().float().numpy(),
        meta,
    )


# ──────────────────────────────────────────────────────────────
# 节点采样（避免节点过多导致可视化过密）
# ──────────────────────────────────────────────────────────────

def sample_nodes(meta: dict, max_nodes: int, rng: np.random.Generator):
    """
    对三类节点分别采样，返回采样后的全局索引和对应标签。

    Returns
    -------
    indices   : np.ndarray, shape (M,)  在完整节点矩阵中的行索引
    type_labels: np.ndarray, shape (M,)  0=attr, 1=obj, 2=comp
    attr_labels: np.ndarray, shape (M,)  对应 attr 的全局索引（-1表示obj节点）
    names     : list[str]               节点名称（用于 hover / 标注）
    """
    na = meta["num_attrs"]
    no = meta["num_objs"]
    nc = meta["num_comps"]

    # 采样各类节点
    def _sample(start, count, max_n):
        idx = np.arange(start, start + count)
        if count > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        return np.sort(idx)

    attr_idx = _sample(0,       na, max_nodes)
    obj_idx  = _sample(na,      no, max_nodes)
    comp_idx = _sample(na + no, nc, max_nodes)

    indices    = np.concatenate([attr_idx, obj_idx, comp_idx])
    type_labels = np.concatenate([
        np.zeros(len(attr_idx),  dtype=int),
        np.ones(len(obj_idx),    dtype=int),
        np.full(len(comp_idx),   2, dtype=int),
    ])

    # 构建 attr_labels：comp 节点标记其对应的 attr 全局索引
    attr_of_comp = np.full(na + no + nc, -1, dtype=int)
    for i, (a, o) in enumerate(meta["train_pairs"]):
        attr_idx_in_vocab = meta["attrs"].index(a)
        attr_of_comp[na + no + i] = attr_idx_in_vocab

    attr_labels = attr_of_comp[indices]

    # 节点名称
    attr_names = list(meta["attrs"])
    obj_names  = list(meta["objs"])
    comp_names = [f"{a} {o}" for a, o in meta["train_pairs"]]
    all_names  = attr_names + obj_names + comp_names
    names      = [all_names[i] for i in indices]

    return indices, type_labels, attr_labels, names


# ──────────────────────────────────────────────────────────────
# 降维
# ──────────────────────────────────────────────────────────────

def reduce_dim(embeddings: np.ndarray, method: str,
               perplexity: float, n_neighbors: int, seed: int):
    """对给定嵌入矩阵做降维，返回 (N, 2) 的 2D 坐标。"""
    if method == "tsne":
        from sklearn.manifold import TSNE
        print(f"    运行 t-SNE (perplexity={perplexity}) ...")
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP 未安装，请运行：pip install umap-learn"
            )
        print(f"    运行 UMAP (n_neighbors={n_neighbors}) ...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"未知降维方法：{method}")


# ──────────────────────────────────────────────────────────────
# 绘图核心
# ──────────────────────────────────────────────────────────────

# 配色方案
NODE_COLORS = {
    0: "#4C9BE8",   # attr  - 蓝色
    1: "#E8764C",   # obj   - 橙色
    2: "#6BBF59",   # comp  - 绿色
}
NODE_MARKERS = {
    0: "^",   # attr  - 三角形
    1: "s",   # obj   - 方形
    2: "o",   # comp  - 圆形
}
NODE_LABELS = {0: "Attribute", 1: "Object", 2: "Composition"}
NODE_SIZES  = {0: 40, 1: 40, 2: 25}


def _scatter_by_type(ax, coords_2d, type_labels, alpha=0.7, title=""):
    """按节点类型着色绘制散点图（四象限图中的两列）。"""
    for t in [2, 1, 0]:   # comp 先画，attr/obj 后画（层次感）
        mask = type_labels == t
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=NODE_COLORS[t],
            marker=NODE_MARKERS[t],
            s=NODE_SIZES[t],
            alpha=alpha,
            linewidths=0.3,
            edgecolors="white",
            label=NODE_LABELS[t],
            zorder=3 - t,
        )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")


def _scatter_by_shared_attr(ax, coords_2d, type_labels, attr_labels,
                             top_k_attrs=8, alpha=0.65, title=""):
    """
    高亮共享属性的节点簇：
      - 选取拥有最多 comp 节点的 top_k_attrs 个属性
      - 对应的 attr 节点和 comp 节点用同色高亮
      - 其余节点画成浅灰色
    """
    # 统计每个 attr 对应的 comp 数量
    attr_comp_count = defaultdict(int)
    comp_mask = type_labels == 2
    for al in attr_labels[comp_mask]:
        if al >= 0:
            attr_comp_count[al] += 1

    top_attrs = sorted(attr_comp_count, key=attr_comp_count.get, reverse=True)
    top_attrs = top_attrs[:top_k_attrs]

    # 调色板（区分度高的颜色序列）
    palette = matplotlib.colormaps.get_cmap("tab10").resampled(top_k_attrs)
    colors  = [palette(i) for i in range(top_k_attrs)]

    # 先画灰色背景节点
    all_highlight = set()
    for rank, attr_id in enumerate(top_attrs):
        # attr 节点本身
        attr_node_mask = (type_labels == 0) & (attr_labels == attr_id)
        # comp 节点
        comp_node_mask = (type_labels == 2) & (attr_labels == attr_id)
        highlight_idx  = np.where(attr_node_mask | comp_node_mask)[0]
        all_highlight.update(highlight_idx.tolist())

    bg_mask = np.ones(len(type_labels), dtype=bool)
    bg_mask[list(all_highlight)] = False
    ax.scatter(
        coords_2d[bg_mask, 0], coords_2d[bg_mask, 1],
        c="#DDDDDD", s=15, alpha=0.4, linewidths=0, zorder=1,
    )

    # 画高亮节点
    for rank, attr_id in enumerate(top_attrs):
        color = colors[rank]
        attr_node_mask = (type_labels == 0) & (attr_labels == attr_id)
        comp_node_mask = (type_labels == 2) & (attr_labels == attr_id)

        # attr 节点（三角形，较大）
        if attr_node_mask.sum() > 0:
            ax.scatter(
                coords_2d[attr_node_mask, 0],
                coords_2d[attr_node_mask, 1],
                c=[color], marker="^", s=120,
                alpha=0.95, linewidths=0.8,
                edgecolors="black", zorder=5,
            )
        # comp 节点（圆形，较小）
        if comp_node_mask.sum() > 0:
            ax.scatter(
                coords_2d[comp_node_mask, 0],
                coords_2d[comp_node_mask, 1],
                c=[color], marker="o", s=30,
                alpha=0.8, linewidths=0.3,
                edgecolors="white", zorder=4,
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")


def plot_four_panel(coords_before, coords_after,
                    type_labels, attr_labels,
                    method_name: str, dataset_name: str,
                    save_path: str, dpi: int):
    """绘制 2×2 四格对比图并保存。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#F8F9FA")

    method_upper = method_name.upper()

    # [0,0] 传播前 - 按类型着色
    _scatter_by_type(
        axes[0, 0], coords_before, type_labels,
        title=f"Before Propagation — Node Type\n({method_upper})",
    )

    # [0,1] 传播后 - 按类型着色
    _scatter_by_type(
        axes[0, 1], coords_after, type_labels,
        title=f"After Propagation — Node Type\n({method_upper})",
    )

    # [1,0] 传播前 - 按共享属性着色
    _scatter_by_shared_attr(
        axes[1, 0], coords_before, type_labels, attr_labels,
        title=f"Before Propagation — Shared Attribute\n({method_upper})",
    )

    # [1,1] 传播后 - 按共享属性着色
    _scatter_by_shared_attr(
        axes[1, 1], coords_after, type_labels, attr_labels,
        title=f"After Propagation — Shared Attribute\n({method_upper})",
    )

    # ── 统一图例（类型） ──
    legend_patches = [
        mpatches.Patch(color=NODE_COLORS[t], label=NODE_LABELS[t])
        for t in [0, 1, 2]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    # ── 总标题 ──
    fig.suptitle(
        f"Node Embedding Visualization — {dataset_name}\n"
        f"Heterogeneous Graph Propagation Effect ({method_upper})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ── 左右列标注（Before / After） ──
    for col, label in enumerate(["Before Propagation", "After Propagation"]):
        fig.text(
            0.25 + col * 0.5, 0.505,
            label,
            ha="center", va="center",
            fontsize=10, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0E0E0", alpha=0.6),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ 已保存：{save_path}")


def plot_single_panel(coords_before, coords_after,
                      type_labels, attr_labels,
                      method_name: str, dataset_name: str,
                      save_dir: str, dpi: int):
    """
    额外输出两张独立的高分辨率单图：
      - type_comparison.png   : 传播前后按类型着色
      - attr_comparison.png   : 传播前后按共享属性着色
    用于论文直接插图（更清晰）。
    """
    method_upper = method_name.upper()

    for mode in ["type", "attr"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor("white")

        for col, (coords, stage) in enumerate([
            (coords_before, "Before Propagation"),
            (coords_after,  "After Propagation"),
        ]):
            ax = axes[col]
            if mode == "type":
                _scatter_by_type(
                    ax, coords, type_labels,
                    title=f"{stage}\n({method_upper})",
                )
            else:
                _scatter_by_shared_attr(
                    ax, coords, type_labels, attr_labels,
                    title=f"{stage}\n({method_upper})",
                )

        # 图例
        if mode == "type":
            legend_patches = [
                mpatches.Patch(color=NODE_COLORS[t], label=NODE_LABELS[t])
                for t in [0, 1, 2]
            ]
            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=3,
                fontsize=11,
                bbox_to_anchor=(0.5, -0.02),
            )

        title_suffix = "Node Type" if mode == "type" else "Shared Attribute Cluster"
        fig.suptitle(
            f"{title_suffix} — {dataset_name}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        fname = f"{method_name}_{mode}_comparison.pdf"
        fpath = os.path.join(save_dir, fname)
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight", format="pdf")

        # 同时保存 PNG
        png_path = fpath.replace(".pdf", ".png")
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 已保存：{fpath}")
        print(f"  ✓ 已保存：{png_path}")


# ──────────────────────────────────────────────────────────────
# 量化聚类质量：Silhouette Score
# ──────────────────────────────────────────────────────────────

def compute_cluster_quality(coords_2d, type_labels):
    """
    用 Silhouette Score 量化按节点类型划分的聚类质量，
    值越高代表三类节点在 2D 空间中分离越好。
    """
    try:
        from sklearn.metrics import silhouette_score
        score = silhouette_score(coords_2d, type_labels, metric="euclidean")
        return score
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    rng   = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── 1. 加载配置与数据集 ──
    print("[1/4] 加载配置与数据集...")
    cfg = load_config(args.config)
    cfg["num_workers"] = 0
    _, train_dataset = get_dataloader(cfg, "train")
    dataset_name = cfg.get("dataset", "dataset")
    print(f"      数据集：{dataset_name}  "
          f"attrs={train_dataset.num_attrs}  "
          f"objs={train_dataset.num_objs}  "
          f"train_pairs={train_dataset.num_train_pairs}")

    # ── 2. 加载模型 ──
    print("[2/4] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RGIPCOL(cfg=cfg, dataset=train_dataset, device=device).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    # ── 3. 提取嵌入 ──
    print("[3/4] 提取节点嵌入...")
    emb_before, emb_after, meta = extract_embeddings(model)
    print(f"      节点总数：{emb_before.shape[0]}  维度：{emb_before.shape[1]}")

    # 采样（控制可视化密度）
    indices, type_labels, attr_labels, names = sample_nodes(
        meta, args.max_nodes, rng
    )
    emb_before_sampled = emb_before[indices]
    emb_after_sampled  = emb_after[indices]
    print(f"      采样后节点数：{len(indices)}  "
          f"(attr={( type_labels==0).sum()}  "
          f"obj={(type_labels==1).sum()}  "
          f"comp={(type_labels==2).sum()})")

    # ── 4. 降维 & 绘图 ──
    methods = (["tsne", "umap"] if args.method == "both" else [args.method])

    for method in methods:
        print(f"\n[4/4] {method.upper()} 降维与可视化...")

        # 将传播前后节点拼接后一起降维（保证坐标系一致，便于对比）
        combined = np.vstack([emb_before_sampled, emb_after_sampled])
        combined_2d = reduce_dim(
            combined, method,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
        n = len(emb_before_sampled)
        coords_before = combined_2d[:n]
        coords_after  = combined_2d[n:]

        # 量化聚类质量
        sil_before = compute_cluster_quality(coords_before, type_labels)
        sil_after  = compute_cluster_quality(coords_after,  type_labels)
        if sil_before is not None:
            print(f"    Silhouette Score — 传播前: {sil_before:.4f}  "
                  f"传播后: {sil_after:.4f}  "
                  f"(提升: {sil_after - sil_before:+.4f})")

        # 主四格图
        four_panel_path = os.path.join(
            args.save_dir,
            f"{dataset_name}_{method}_4panel.png",
        )
        plot_four_panel(
            coords_before, coords_after,
            type_labels, attr_labels,
            method_name=method,
            dataset_name=dataset_name,
            save_path=four_panel_path,
            dpi=args.dpi,
        )

        # 独立单图（适合论文直接插入）
        plot_single_panel(
            coords_before, coords_after,
            type_labels, attr_labels,
            method_name=method,
            dataset_name=dataset_name,
            save_dir=args.save_dir,
            dpi=args.dpi,
        )

    print("\n✓ 全部可视化完成！")
    print(f"  输出目录：{os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()