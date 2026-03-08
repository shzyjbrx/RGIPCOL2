"""
composition_dataset.py
----------------------
支持标准 CZSL compositional-split-natural 格式：

    data_dir/
    ├── compositional-split-natural/
    │   ├── train_pairs.txt    每行：attr obj
    │   ├── val_pairs.txt
    │   └── test_pairs.txt
    └── images/
        └── <obj>/
            └── <attr>_<obj>_NNNNN.jpg   (MIT-States 命名规范)

同时兼容旧版 metadata pkl 格式（自动检测）。
"""

import os
from glob import glob
from itertools import product
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _read_pairs(path: str) -> List[Tuple[str, str]]:
    """
    读取 pairs txt 文件。
    每行格式：  attr obj
    返回 list of (attr, obj)
    """
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


# def _scan_images(
#     images_dir: str,
#     valid_pairs: set,
#     attr2idx: dict,
#     obj2idx: dict,
# ) -> List[Tuple[str, str, str]]:
#     """
#     扫描 images/ 目录，返回属于 valid_pairs 的图像列表。

#     MIT-States 目录结构：
#         images/<obj>/<attr>_<obj>_NNNNN.jpg
#     图像文件名解析规则：
#         文件名首个下划线前的部分 → attr（已知情况下，用 attr2idx 校验）
#         父文件夹名 → obj

#     返回 list of (full_img_path, attr, obj)
#     """
#     records = []
#     obj_dirs = [d for d in os.listdir(images_dir)
#                 if os.path.isdir(os.path.join(images_dir, d))]

#     for obj in obj_dirs:
#         if obj not in obj2idx:
#             continue
#         obj_dir = os.path.join(images_dir, obj)
#         for fname in os.listdir(obj_dir):
#             if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue
#             # 从文件名解析 attr：取第一个 "_" 之前的部分
#             # MIT-States 文件名示例：sliced_apple_000001.jpg → attr=sliced
#             attr = fname.split("_")[0]
#             if attr not in attr2idx:
#                 continue
#             if (attr, obj) not in valid_pairs:
#                 continue
#             full_path = os.path.join(obj_dir, fname)
#             records.append((full_path, attr, obj))

#     return records
def _scan_images(images_dir, valid_pairs, attr2idx, obj2idx):
    records = []
    comp_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    for comp in comp_dirs:
        # 假设文件夹名字为 "attr obj" 或 "attr_obj"
        parts = comp.replace("_", " ").split()
        if len(parts) < 2:
            continue
        attr, obj = parts[0], parts[1]
        
        # 校验 attr 和 obj 是否在词汇表且属于当前 split
        if attr not in attr2idx or obj not in obj2idx:
            continue
        if (attr, obj) not in valid_pairs:
            continue
            
        comp_dir = os.path.join(images_dir, comp)
        for fname in os.listdir(comp_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full_path = os.path.join(comp_dir, fname)
            records.append((full_path, attr, obj))
            
    return records


# ──────────────────────────────────────────────────────────────
# 主 Dataset 类
# ──────────────────────────────────────────────────────────────

class CompositionDataset(Dataset):
    """
    通用 CZSL Dataset，优先使用 compositional-split-natural txt 格式。

    Parameters
    ----------
    data_dir  : 数据集根目录
    split     : "train" | "val" | "test"
    transform : CLIP preprocess（torchvision transform）
    """

    def __init__(self, data_dir: str, split: str, transform=None):
        super().__init__()
        assert split in ("train", "val", "test")

        self.data_dir  = data_dir
        self.split     = split
        self.transform = transform

        # ── 加载组合对与词汇表 ──
        self._load_metadata()

        # ── 加载当前 split 的图像列表 ──
        self._load_images(split)

    # ────────────────────────────
    # 元数据加载（优先 txt，备选 pkl）
    # ────────────────────────────

    def _load_metadata(self):
        # ── 优先尝试 compositional-split-natural txt 格式 ──
        split_dir = os.path.join(self.data_dir, "compositional-split-natural")
        train_txt = os.path.join(split_dir, "train_pairs.txt")

        if os.path.exists(train_txt):
            self._load_from_txt(split_dir)
        else:
            # 回退到 pkl
            self._load_from_pkl()

    def _load_from_txt(self, split_dir: str):
        """从 compositional-split-natural/*.txt 加载"""
        print(f"[Dataset] 使用 compositional-split-natural txt 格式")

        self.train_pairs = _read_pairs(os.path.join(split_dir, "train_pairs.txt"))
        self.val_pairs   = _read_pairs(os.path.join(split_dir, "val_pairs.txt"))
        self.test_pairs  = _read_pairs(os.path.join(split_dir, "test_pairs.txt"))

        # 从三个 split 的所有 pair 中提取词汇表
        all_attrs = set()
        all_objs  = set()
        for (a, o) in self.train_pairs + self.val_pairs + self.test_pairs:
            all_attrs.add(a)
            all_objs.add(o)

        self.attrs = sorted(all_attrs)
        self.objs  = sorted(all_objs)
        self._build_indices()

    def _load_from_pkl(self):
        """从 pkl 元数据文件加载（旧版兼容）"""
        import pickle
        candidates = [
            os.path.join(self.data_dir, "metadata_compositional_split_natural.t7"),
            os.path.join(self.data_dir, "metadata.pkl"),
        ]
        meta_path = next((p for p in candidates if os.path.exists(p)), None)
        if meta_path is None:
            raise FileNotFoundError(
                f"[Dataset] 未找到元数据文件，已尝试：\n"
                f"  {candidates[0]}\n  {candidates[1]}\n"
                f"  {os.path.join(self.data_dir, 'compositional-split-natural/train_pairs.txt')}"
            )
        print(f"[Dataset] 使用 pkl 元数据：{meta_path}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        self.attrs       = sorted(list(meta["attrs"]))
        self.objs        = sorted(list(meta["objs"]))
        self.train_pairs = [(a, o) for a, o in meta["train_pairs"]]
        self.val_pairs   = [(a, o) for a, o in meta["val_pairs"]]
        self.test_pairs  = [(a, o) for a, o in meta["test_pairs"]]
        self._build_indices()

    def _build_indices(self):
        """构建词汇表索引和各类候选集"""
        self.attr2idx = {a: i for i, a in enumerate(self.attrs)}
        self.obj2idx  = {o: i for i, o in enumerate(self.objs)}

        # seen = train 对
        self.seen_pairs = self.train_pairs

        # unseen = test 中不在 train 里的对
        seen_set          = set(map(tuple, self.seen_pairs))
        # self.unseen_pairs = [p for p in self.test_pairs if tuple(p) not in seen_set]
        if self.split == "val":
            self.unseen_pairs = [p for p in self.val_pairs if tuple(p) not in seen_set]
        else:
            self.unseen_pairs = [p for p in self.test_pairs if tuple(p) not in seen_set]

        # 闭合世界候选集 = seen ∪ unseen
        closed_set         = seen_set | set(map(tuple, self.unseen_pairs))
        self.closed_pairs  = sorted(list(closed_set))

        # 开放世界候选集 = 全笛卡尔积
        self.all_pairs = [(a, o) for a, o in product(self.attrs, self.objs)]

        # 索引查找表
        self.train_pair2idx = {tuple(p): i for i, p in enumerate(self.train_pairs)}
        self.all_pair2idx   = {tuple(p): i for i, p in enumerate(self.all_pairs)}

    # ────────────────────────────
    # 图像列表加载
    # ────────────────────────────

    def _load_images(self, split: str):
        """
        扫描 images/ 目录，按当前 split 的 pair 集合过滤图像。
        """
        images_dir = os.path.join(self.data_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"[Dataset] images 目录不存在：{images_dir}"
            )

        # 当前 split 对应的 pair 集合
        if split == "train":
            valid_pairs = set(map(tuple, self.train_pairs))
        elif split == "val":
            valid_pairs = set(map(tuple, self.val_pairs))
        else:
            valid_pairs = set(map(tuple, self.test_pairs))

        print(f"[Dataset] 扫描图像目录：{images_dir}")
        print(f"[Dataset] split={split}，有效 pair 数={len(valid_pairs)}")

        records = _scan_images(images_dir, valid_pairs, self.attr2idx, self.obj2idx)

        if len(records) == 0:
            raise RuntimeError(
                f"[Dataset] {split} split 中未找到任何图像！\n"
                f"  images_dir: {images_dir}\n"
                f"  valid_pairs 示例: {list(valid_pairs)[:5]}\n"
                "请检查图像文件命名是否为 <attr>_<obj>_NNNNN.jpg 格式"
            )

        self.data = records
        print(f"[Dataset] {split} split 加载完成，共 {len(self.data)} 张图像")

    # ────────────────────────────
    # Dataset 接口
    # ────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, attr, obj = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        attr_idx = self.attr2idx[attr]
        obj_idx  = self.obj2idx[obj]
        pair     = (attr, obj)

        if self.split == "train":
            pair_idx = self.train_pair2idx.get(pair, -1)
        else:
            pair_idx = self.all_pair2idx.get(pair, -1)

        return {
            "image"   : image,
            "attr_idx": attr_idx,
            "obj_idx" : obj_idx,
            "pair_idx": pair_idx,
            "attr"    : attr,
            "obj"     : obj,
        }

    # ────────────────────────────
    # 便捷属性
    # ────────────────────────────

    @property
    def num_attrs(self):
        return len(self.attrs)

    @property
    def num_objs(self):
        return len(self.objs)

    @property
    def num_train_pairs(self):
        return len(self.train_pairs)

    def __repr__(self):
        return (
            f"CompositionDataset(split={self.split}, "
            f"#attrs={self.num_attrs}, #objs={self.num_objs}, "
            f"#train_pairs={self.num_train_pairs}, "
            f"#samples={len(self.data)})"
        )