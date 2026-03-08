"""
generate_node_descriptions.py
------------------------------
使用 Qwen2.5-7B-Instruct 为 CZSL 数据集的三类节点生成细粒度涌现描述：

  1. attr 节点  : 属性的视觉表现描述（如 "sliced" → 截面纹理、边缘形态）
  2. obj  节点  : 物体的外观与材质描述（如 "apple" → 颜色、形状、表面纹理）
  3. comp 节点  : 属性施加于物体后的涌现视觉描述（如 "sliced apple" → 白色果肉、星形核）

输出格式（统一的 JSON 文件）：
    {
        "sliced":        "Sliced refers to a clean cut surface revealing ...",
        "apple":         "An apple is a round fruit with smooth red or green skin ...",
        "sliced apple":  "A sliced apple exposes white moist flesh with a star-shaped core ...",
        ...
    }

使用方法：
    # 生成 MIT-States 的三类节点描述
    python generate_node_descriptions.py \
        --data_root  ./data/mit-states \
        --output     ./LLM_descriptions/mit_node_descriptions.json \
        --model_id   Qwen/Qwen2.5-7B-Instruct

    # 指定只生成某类节点（断点续跑时有用）
    python generate_node_descriptions.py \
        --data_root  ./data/mit-states \
        --output     ./LLM_descriptions/mit_node_descriptions.json \
        --node_types attr obj comp

注意：
    - 输出文件与 generate_edge_weights.py 的输出是独立的两个 JSON
    - llm_feature_loader.py 的 load_node_descriptions() 会读取此文件
    - 若已有部分结果，脚本会自动断点续跑（跳过已有 key）
    - 包含中文检测与重试机制
"""

import os
import re
import json
import argparse
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate LLM node descriptions for CZSL")
    p.add_argument("--data_root",   type=str, default="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/mit-states",
                   help="数据集根目录，如 ./data/mit-states")
    p.add_argument("--output",      type=str, default="./LLM/mit_edge_weights.json",
                   help="输出 JSON 文件路径，如 ./LLM_descriptions/mit_node_descriptions.json")
    p.add_argument("--model_id",    type=str, default="Qwen/Qwen2.5-7B-Instruct",
                   help="Huggingface 模型 ID 或本地路径")
    p.add_argument("--node_types",  type=str, nargs="+",
                   default=["attr", "obj", "comp"],
                   choices=["attr", "obj", "comp"],
                   help="需要生成的节点类型（默认全部）")
    p.add_argument("--max_retries", type=int, default=3,
                   help="遇到中文输出时的最大重试次数")
    p.add_argument("--max_new_tokens", type=int, default=80,
                   help="每次生成的最大 token 数（控制描述长度）")
    return p.parse_args()


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_vocab(data_root: str):
    """
    从 compositional-split-natural 目录读取全部 attrs、objs、pairs。
    返回去重后的有序列表。
    """
    split_dir = os.path.join(data_root, "compositional-split-natural")
    attrs_set = set()
    objs_set  = set()
    pairs_set = set()

    for fname in ["train_pairs.txt", "val_pairs.txt", "test_pairs.txt"]:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    a, o = parts[0], parts[1]
                    attrs_set.add(a)
                    objs_set.add(o)
                    pairs_set.add((a, o))

    attrs = sorted(attrs_set)
    objs  = sorted(objs_set)
    pairs = sorted(pairs_set)

    print(f"[Vocab] 加载完成：{len(attrs)} 个属性，{len(objs)} 个物体，{len(pairs)} 个组合对")
    return attrs, objs, pairs


def clean_name(name: str) -> str:
    """将下划线/点替换为空格，用于 prompt 中的可读名称"""
    return name.replace("_", " ").replace(".", " ").strip()


# ─────────────────────────────────────────────
# Prompt 构建（三类节点各有专属模板）
# ─────────────────────────────────────────────

def build_attr_prompt(attr: str) -> list:
    """
    属性节点 Prompt：描述该属性对物体产生的视觉变化特征。
    重点：纹理、形变、颜色、边缘等视觉层面的物理效果。
    """
    attr_clean = clean_name(attr)
    system = (
        "You are a concise English computer vision expert. "
        "Output only English. No Chinese. No bullet points."
    )
    user = (
        f"Describe the visual appearance of the attribute '{attr_clean}' "
        f"when it is applied to physical objects. "
        f"Focus on: texture changes, shape deformation, color shifts, "
        f"or surface effects this attribute typically causes. "
        f"Write one sentence only."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def build_obj_prompt(obj: str) -> list:
    """
    物体节点 Prompt：描述物体的基础外观与材质特征。
    重点：颜色、形状、表面纹理、材质、典型尺寸。
    """
    obj_clean = clean_name(obj)
    system = (
        "You are a concise English computer vision expert. "
        "Output only English. No Chinese. No bullet points."
    )
    user = (
        f"Describe the typical visual appearance of '{obj_clean}' "
        f"as it would appear in a photograph. "
        f"Focus on: shape, color, surface texture, and material. "
        f"Write one sentence only."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def build_comp_prompt(attr: str, obj: str) -> list:
    """
    组合节点 Prompt：描述属性作用于物体后产生的涌现视觉特征。
    重点：不可由属性和物体简单叠加推导出来的"新"视觉细节。
    这是最关键的一类，与之前 mit_qwen.json 的生成脚本对应。
    """
    attr_clean = clean_name(attr)
    obj_clean  = clean_name(obj)
    system = (
        "You are an English-speaking computer vision assistant. "
        "You strictly output English text only. "
        "Do NOT use any Chinese characters or non-English scripts."
    )
    user = (
        f"Describe the emergent visual features of '{attr_clean} {obj_clean}'. "
        f"Focus on physical changes (e.g. texture, deformation, color) "
        f"caused by '{attr_clean}' acting on '{obj_clean}'. "
        f"Constraint: Output a single short sentence in English. No Chinese allowed."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


# ─────────────────────────────────────────────
# 中文检测
# ─────────────────────────────────────────────

def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# ─────────────────────────────────────────────
# 单条生成（含重试）
# ─────────────────────────────────────────────

def generate_one(
    tokenizer,
    model,
    messages     : list,
    max_new_tokens: int,
    max_retries  : int,
    key_for_log  : str,
) -> str:
    """生成单条描述，带中文检测与重试"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    final_response = ""
    for attempt in range(max_retries):
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens = max_new_tokens,
                temperature    = 0.7 + attempt * 0.1,
                top_p          = 0.9,
                do_sample      = True,
            )
        out_ids = [
            o[len(i):] for i, o in
            zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(
            out_ids, skip_special_tokens=True
        )[0].strip()

        if not contains_chinese(response):
            final_response = response
            break
        else:
            if attempt == max_retries - 1:
                print(f"\n[!] '{key_for_log}' 在 {max_retries} 次重试后仍含中文，保留原样")
                final_response = response

    return final_response


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── 加载词汇表 ──
    attrs, objs, pairs = load_vocab(args.data_root)

    # ── 断点续跑：加载已有结果 ──
    results = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                print(f"[Resume] 断点续跑，已有 {len(results)} 条描述")
            except json.JSONDecodeError:
                print("[Resume] JSON 解析失败，从零开始")

    # ── 加载模型 ──
    print(f"\n[Model] 加载模型：{args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
    )
    print(f"[Model] 加载完成\n")

    def save():
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_generated = 0

    # ──────────────────────────────────────────
    # 1. attr 节点描述
    # ──────────────────────────────────────────
    if "attr" in args.node_types:
        print(f"[Stage 1/3] 生成 attr 节点描述（共 {len(attrs)} 个）...")
        for attr in tqdm(attrs, desc="attr nodes"):
            key = clean_name(attr)
            if key in results and not contains_chinese(results[key]):
                continue   # 已有合法结果，跳过

            messages = build_attr_prompt(attr)
            desc = generate_one(
                tokenizer, model, messages,
                args.max_new_tokens, args.max_retries,
                key_for_log=f"attr:{attr}"
            )
            results[key] = desc
            total_generated += 1

            if total_generated % 50 == 0:
                save()
                print(f"  [Auto-save] 已保存 {len(results)} 条")

    # ──────────────────────────────────────────
    # 2. obj 节点描述
    # ──────────────────────────────────────────
    if "obj" in args.node_types:
        print(f"\n[Stage 2/3] 生成 obj 节点描述（共 {len(objs)} 个）...")
        for obj in tqdm(objs, desc="obj nodes"):
            key = clean_name(obj)
            if key in results and not contains_chinese(results[key]):
                continue

            messages = build_obj_prompt(obj)
            desc = generate_one(
                tokenizer, model, messages,
                args.max_new_tokens, args.max_retries,
                key_for_log=f"obj:{obj}"
            )
            results[key] = desc
            total_generated += 1

            if total_generated % 50 == 0:
                save()

    # ──────────────────────────────────────────
    # 3. comp 节点描述（最重要，数量最多）
    # ──────────────────────────────────────────
    if "comp" in args.node_types:
        print(f"\n[Stage 3/3] 生成 comp 节点描述（共 {len(pairs)} 个）...")
        for (attr, obj) in tqdm(pairs, desc="comp nodes"):
            key = f"{clean_name(attr)} {clean_name(obj)}"
            if key in results and not contains_chinese(results[key]):
                continue

            messages = build_comp_prompt(attr, obj)
            desc = generate_one(
                tokenizer, model, messages,
                args.max_new_tokens, args.max_retries,
                key_for_log=f"comp:{attr} {obj}"
            )
            results[key] = desc
            total_generated += 1

            if total_generated % 50 == 0:
                save()

    # ── 最终保存 ──
    save()
    print(f"\n[✓] 全部完成！")
    print(f"    本次新生成：{total_generated} 条")
    print(f"    文件总条数：{len(results)} 条")
    print(f"    保存路径：{args.output}")

    # ── 统计输出长度分布 ──
    lengths = [len(v.split()) for v in results.values()]
    import numpy as np
    print(f"\n[Stats] 描述长度（词数）：")
    print(f"    min={min(lengths)}  max={max(lengths)}  "
          f"mean={np.mean(lengths):.1f}  median={np.median(lengths):.1f}")

    # ── 检查仍有中文的条目 ──
    bad = [(k, v) for k, v in results.items() if contains_chinese(v)]
    if bad:
        print(f"\n[!] 仍有 {len(bad)} 条含中文，建议手动检查或删除后重跑：")
        for k, v in bad[:5]:
            print(f"    '{k}': {v[:50]}...")
    else:
        print(f"\n[✓] 所有描述均为纯英文")


if __name__ == "__main__":
    main()