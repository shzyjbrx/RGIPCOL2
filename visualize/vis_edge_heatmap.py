"""
visualize/vis_edge_heatmap.py
-----------------------------
LLM (attr, obj) 适用性矩阵热力图可视化

生成一张以属性为行、物体为列的热力图，
颜色深浅反映 LLM 评分的物理适用性高低（0=不可行，1=高度自然）。

同时生成辅助统计图（可选）：
  - 每个属性的平均适用性柱状图（行均值）
  - 每个物体被各属性修饰的平均适用性（列均值）

用法：
    python visualize/vis_edge_heatmap.py \
        --weights_path  ./LLM_descriptions/mit_edge_weights.json \
        --data_root     ./data/mit-states \
        --output        visualize/output/edge_heatmap_mit.pdf \
        --top_attrs     30 \
        --top_objs      40 \
        --sort_by       variance        # 优先展示差异大的行/列
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_path", type=str, required=True,
                   help="generate_edge_weights.py 的输出 JSON")
    p.add_argument("--data_root",    type=str, required=True,
                   help="数据集根目录（用于加载词汇表顺序）")
    p.add_argument("--output",       type=str,
                   default="visualize/output/edge_heatmap.pdf")
    p.add_argument("--top_attrs",    type=int, default=30,
                   help="展示适用性方差最大的前 N 个属性")
    p.add_argument("--top_objs",     type=int, default=40,
                   help="展示适用性方差最大的前 N 个物体")
    p.add_argument("--sort_by",      type=str, default="variance",
                   choices=["variance", "mean", "alphabetical"],
                   help="行列排序依据")
    p.add_argument("--annotate_top", type=int, default=5,
                   help="在热力图上标注出分数最高/最低的前 N 个 cell")
    return p.parse_args()


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_vocab(data_root: str):
    split_dir = os.path.join(data_root, "compositional-split-natural")
    attrs_s, objs_s = set(), set()
    for fname in ["train_pairs.txt", "val_pairs.txt", "test_pairs.txt"]:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    attrs_s.add(parts[0])
                    objs_s.add(parts[1])
    return sorted(attrs_s), sorted(objs_s)


def build_matrix(weights: dict, attrs: list, objs: list) -> np.ndarray:
    """
    构建 (num_attrs, num_objs) 适用性矩阵。
    缺失的 (attr, obj) 对用 NaN 填充（在热力图中显示为灰色）。
    """
    A, O = len(attrs), len(objs)
    mat  = np.full((A, O), np.nan, dtype=float)

    attr2i = {a: i for i, a in enumerate(attrs)}
    obj2i  = {o: i for i, o in enumerate(objs)}

    for key, score in weights.items():
        parts = key.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        a_clean = parts[0].replace(" ", "_")
        o_clean = parts[1].replace(" ", "_")
        # 尝试不同格式匹配
        a_idx = attr2i.get(a_clean) or attr2i.get(parts[0])
        o_idx = obj2i.get(o_clean)  or obj2i.get(parts[1])
        if a_idx is not None and o_idx is not None:
            mat[a_idx, o_idx] = float(score)

    coverage = (~np.isnan(mat)).sum() / (A * O) * 100
    print(f"[Heatmap] 矩阵覆盖率：{coverage:.1f}%  "
          f"({(~np.isnan(mat)).sum()} / {A*O} 个 pair)")
    return mat


# ──────────────────────────────────────────────
# 选取最具信息量的行列
# ──────────────────────────────────────────────

def select_rows_cols(mat, attrs, objs, top_attrs, top_objs, sort_by):
    """按 sort_by 标准选出 top_attrs 行和 top_objs 列"""
    def score_rows(m):
        if sort_by == "variance":
            return np.nanvar(m, axis=1)
        elif sort_by == "mean":
            return np.nanmean(m, axis=1)
        else:
            return np.arange(m.shape[0])  # alphabetical = original order

    def score_cols(m):
        if sort_by == "variance":
            return np.nanvar(m, axis=0)
        elif sort_by == "mean":
            return np.nanmean(m, axis=0)
        else:
            return np.arange(m.shape[1])

    row_scores = score_rows(mat)
    col_scores = score_cols(mat)

    # 按分数降序取 top N
    if sort_by == "alphabetical":
        row_idx = np.arange(min(top_attrs, len(attrs)))
        col_idx = np.arange(min(top_objs,  len(objs)))
    else:
        row_idx = np.argsort(row_scores)[::-1][:top_attrs]
        col_idx = np.argsort(col_scores)[::-1][:top_objs]

    # 再按均值排序（让热力图更有规律）
    sub = mat[np.ix_(row_idx, col_idx)]
    row_order = np.argsort(np.nanmean(sub, axis=1))[::-1]
    col_order = np.argsort(np.nanmean(sub, axis=0))[::-1]
    row_idx   = row_idx[row_order]
    col_idx   = col_idx[col_order]

    return row_idx, col_idx


# ──────────────────────────────────────────────
# 绘制热力图
# ──────────────────────────────────────────────

def draw_heatmap(mat_sub, row_labels, col_labels, output_path,
                 annotate_top, dataset_name):
    A, O = mat_sub.shape

    # ── 自定义色图：白(0) → 浅黄 → 深蓝(1) ──
    colors_list = [
        (0.95, 0.95, 0.95),   # 0.0 → 浅灰（低适用性）
        (1.00, 0.95, 0.70),   # 0.3 → 淡黄
        (0.30, 0.65, 0.85),   # 0.7 → 天蓝
        (0.10, 0.25, 0.65),   # 1.0 → 深蓝（高适用性）
    ]
    cmap = LinearSegmentedColormap.from_list("llm_weight", colors_list)
    # NaN 显示为中等灰
    cmap.set_bad(color="#C8C8C8")

    # ── 布局：主热力图 + 行列均值条形图 ──
    fig = plt.figure(figsize=(
        max(10, O * 0.32 + 3),
        max(8,  A * 0.28 + 3)
    ))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios  = [O, 2],
        height_ratios = [A, 2],
        hspace=0.05, wspace=0.05
    )
    ax_main  = fig.add_subplot(gs[0, 0])
    ax_col   = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_row   = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_cbar  = fig.add_axes([0.88, 0.35, 0.015, 0.45])

    # ── 主热力图 ──
    im = ax_main.imshow(
        mat_sub, cmap=cmap, vmin=0, vmax=1,
        aspect="auto", interpolation="nearest"
    )

    # 行列标签
    ax_main.set_yticks(range(A))
    ax_main.set_yticklabels(row_labels, fontsize=max(5, 9 - A // 10))
    ax_main.set_xticks(range(O))
    ax_main.set_xticklabels(col_labels, fontsize=max(4, 8 - O // 10),
                            rotation=45, ha="right")
    ax_main.set_ylabel("Attribute", fontsize=11)
    ax_main.set_xlabel("")
    ax_main.set_title(
        f"LLM-Generated Physical Applicability Weights\n"
        f"(attr × obj, dataset: {dataset_name})",
        fontsize=12, fontweight="bold", pad=8
    )

    # ── 标注极值 cell ──
    if annotate_top > 0:
        flat = mat_sub.flatten()
        valid_mask = ~np.isnan(flat)
        if valid_mask.sum() > 0:
            valid_vals  = flat[valid_mask]
            valid_idx   = np.where(valid_mask)[0]
            top_high_i  = valid_idx[np.argsort(valid_vals)[-annotate_top:]]
            top_low_i   = valid_idx[np.argsort(valid_vals)[:annotate_top]]
            for idx in list(top_high_i) + list(top_low_i):
                r, c = divmod(idx, O)
                v    = mat_sub[r, c]
                color = "white" if v > 0.5 else "black"
                ax_main.text(c, r, f"{v:.2f}",
                             ha="center", va="center",
                             fontsize=6, color=color, fontweight="bold")

    # ── 列均值条形图（每个 obj 被属性修饰的平均适用性）──
    col_means = np.nanmean(mat_sub, axis=0)
    ax_col.bar(range(O), col_means, color="#4C9BE8", alpha=0.7, width=0.7)
    ax_col.set_ylim(0, 1.05)
    ax_col.set_ylabel("Mean\n(obj)", fontsize=8)
    ax_col.set_xlabel("Object", fontsize=10)
    ax_col.axhline(np.nanmean(col_means), color="red",
                   linestyle="--", linewidth=0.8, alpha=0.7)
    ax_col.tick_params(axis="x", labelbottom=False)
    ax_col.tick_params(axis="y", labelsize=7)
    ax_col.spines[["top", "right"]].set_visible(False)

    # ── 行均值条形图（每个 attr 对所有物体的平均适用性）──
    row_means = np.nanmean(mat_sub, axis=1)
    ax_row.barh(range(A), row_means, color="#F28C28", alpha=0.7, height=0.7)
    ax_row.set_xlim(0, 1.05)
    ax_row.set_xlabel("Mean\n(attr)", fontsize=8)
    ax_row.axvline(np.nanmean(row_means), color="red",
                   linestyle="--", linewidth=0.8, alpha=0.7)
    ax_row.tick_params(axis="y", labelleft=False)
    ax_row.tick_params(axis="x", labelsize=7)
    ax_row.invert_yaxis()
    ax_row.spines[["top", "left"]].set_visible(False)

    # ── 颜色条 ──
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Applicability Score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[Heatmap] 已保存：{output_path}")

    # ── 打印统计信息 ──
    all_vals = mat_sub[~np.isnan(mat_sub)]
    print(f"\n[Heatmap] 适用性分数统计（选取子矩阵）：")
    print(f"  均值 = {all_vals.mean():.3f}")
    print(f"  方差 = {all_vals.var():.3f}")
    print(f"  最高：{all_vals.max():.3f} | 最低：{all_vals.min():.3f}")
    print(f"  高适用性(>0.8)占比：{(all_vals > 0.8).mean()*100:.1f}%")
    print(f"  低适用性(<0.2)占比：{(all_vals < 0.2).mean()*100:.1f}%")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── 加载数据 ──
    with open(args.weights_path, "r") as f:
        weights = json.load(f)
    print(f"[Heatmap] 加载边权重：{len(weights)} 条")

    attrs, objs = load_vocab(args.data_root)
    mat         = build_matrix(weights, attrs, objs)
    dataset_name = os.path.basename(args.data_root)

    # ── 选取子矩阵 ──
    row_idx, col_idx = select_rows_cols(
        mat, attrs, objs,
        args.top_attrs, args.top_objs,
        args.sort_by
    )
    mat_sub    = mat[np.ix_(row_idx, col_idx)]
    row_labels = [attrs[i].replace("_", " ") for i in row_idx]
    col_labels = [objs[i].replace("_",  " ") for i in col_idx]

    print(f"[Heatmap] 子矩阵大小：{mat_sub.shape[0]} attrs × {mat_sub.shape[1]} objs")

    draw_heatmap(mat_sub, row_labels, col_labels,
                 args.output, args.annotate_top, dataset_name)


if __name__ == "__main__":
    main()