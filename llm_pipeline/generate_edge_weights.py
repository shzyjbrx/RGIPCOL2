"""
generate_edge_weights.py
------------------------
使用 Qwen2.5-7B-Instruct 为每个 (attr, obj) 对生成
"属性对物体的物理适用性分数"，作为 RGCN 中
ATTR2OBJ / OBJ2ATTR 边的权重。

输出格式（JSON）：
    {
        "sliced apple": 0.95,
        "broken glass": 0.98,
        "broken air":   0.05,
        ...
    }

使用方法：
    python generate_edge_weights.py --dataset mit-states --output ./LLM/mit_edge_weights.json
"""

import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/mit-states")
    p.add_argument("--output",      type=str, default="./LLM/mit_edge_weights.json")
    p.add_argument("--model_id",    type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_retries", type=int, default=3)
    return p.parse_args()


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_all_pairs(data_root):
    """从 compositional-split-natural 中加载所有唯一 (attr, obj) 对"""
    split_dir = os.path.join(data_root, "compositional-split-natural")
    pairs = set()
    for fname in ["train_pairs.txt", "val_pairs.txt", "test_pairs.txt"]:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.add((parts[0], parts[1]))
    return sorted(list(pairs))


def clean_text(text):
    return text.replace("_", " ").replace(".", " ").strip()


# ─────────────────────────────────────────────
# Prompt 构建
# ─────────────────────────────────────────────
def construct_prompt(attr: str, obj: str) -> list:
    """
    要求模型输出一个 0-10 的整数分数，衡量属性对物体的物理适用性。
    注意：必须仅输出数字，不得有其他内容。
    """
    attr_clean = clean_text(attr)
    obj_clean  = clean_text(obj)

    system_content = (
        "You are a concise physical reasoning assistant. "
        "You only output a single integer from 0 to 10. "
        "No explanation, no punctuation, just the number."
    )
    user_content = (
        f"Rate the physical applicability of the attribute '{attr_clean}' "
        f"to the object '{obj_clean}' on a scale from 0 to 10.\n"
        f"0 = physically impossible (e.g., 'broken air'), "
        f"10 = highly natural (e.g., 'broken glass').\n"
        f"Output only a single integer."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────
# 解析 LLM 输出为 0-1 float
# ─────────────────────────────────────────────
def parse_score(response: str) -> float:
    """从 LLM 输出字符串中提取数字并归一化到 0-1"""
    # 提取第一个整数
    nums = re.findall(r'\d+', response.strip())
    if not nums:
        return 0.5   # 默认中性分数
    score = int(nums[0])
    score = max(0, min(10, score))   # 截断到 [0, 10]
    return score / 10.0


def contains_invalid(response: str) -> bool:
    """检测回答中是否包含非数字内容（如中文或长文本）"""
    if re.search(r'[\u4e00-\u9fff]', response):
        return True
    if len(response.strip()) > 10:      # 超过10字符说明没有严格遵循指令
        return True
    return False


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── 加载模型 ──
    print(f"[*] 加载模型：{args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── 加载已有结果（断点续跑） ──
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            try:
                results = json.load(f)
                print(f"[*] 断点续跑，已有 {len(results)} 条结果")
            except json.JSONDecodeError:
                pass

    # ── 加载所有 pair ──
    all_pairs = load_all_pairs(args.data_root)
    print(f"[*] 共 {len(all_pairs)} 个唯一 (attr, obj) pair")

    # ── 逐 pair 生成分数 ──
    for attr, obj in tqdm(all_pairs, desc="Generating edge weights"):
        key = f"{attr} {obj}"
        if key in results:
            continue   # 已有结果，跳过

        messages = construct_prompt(attr, obj)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        final_score = 0.5   # 默认值
        for attempt in range(args.max_retries):
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens = 8,           # 只需要一个数字
                    temperature    = 0.1 + attempt * 0.2,
                    top_p          = 0.9,
                    do_sample      = True,
                    pad_token_id   = tokenizer.eos_token_id,  # <--- 加上这行，消除控制台满屏的 pad_token 警告
                )
            out_ids = [
                o[len(i):] for i, o in
                zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(
                out_ids, skip_special_tokens=True
            )[0].strip()

            if not contains_invalid(response):
                final_score = parse_score(response)
                break
            elif attempt == args.max_retries - 1:
                print(f"\n[!] 无法获得合法分数，key='{key}'，原始输出='{response}'，使用默认 0.5")
                final_score = 0.5

        results[key] = final_score

        # 每 100 条保存一次
        if len(results) % 100 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 最终保存 ──
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] 完成！共生成 {len(results)} 条边权重，保存到：{args.output}")

    # 打印统计
    import numpy as np
    scores = list(results.values())
    print(f"    分数分布：min={min(scores):.2f}  max={max(scores):.2f}  "
          f"mean={np.mean(scores):.2f}  std={np.std(scores):.2f}")


if __name__ == "__main__":
    main()