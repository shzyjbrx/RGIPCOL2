"""
llm_pipeline/1_generate_candidates.py
-------------------------------------
第一步：利用本地开源 LLM (如 Qwen2.5) 为 CZSL 三类节点生成 K 个细粒度候选描述。
包含中文检测、JSON 列表解析与自动重试机制。
"""

import os
import re
import sys
import json
import argparse
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate K candidate descriptions using local LLM")
    p.add_argument("--data_root",      type=str, default="./data/mit-states")
    p.add_argument("--save_path",      type=str, default="./LLM/mit_candidate_descriptions.json")
    p.add_argument("--model_id",       type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--node_types",     type=str, nargs="+", default=["attr", "obj", "comp"],
                   choices=["attr", "obj", "comp"])
    p.add_argument("--k_candidates",   type=int, default=3)
    p.add_argument("--max_retries",    type=int, default=6,   # 增加重试次数
                   help="生成失败时的最大重试次数")
    p.add_argument("--max_new_tokens", type=int, default=384)
    return p.parse_args()

# ─────────────────────────────────────────────
# 数据加载与清洗
# ─────────────────────────────────────────────
def load_vocab(data_root: str):
    split_dir = os.path.join(data_root, "compositional-split-natural")
    attrs_set, objs_set, pairs_set = set(), set(), set()
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
    return sorted(attrs_set), sorted(objs_set), sorted(pairs_set)

def clean_name(name: str) -> str:
    return name.replace("_", " ").replace(".", " ").strip()

# ─────────────────────────────────────────────
# Prompt 构建 —— 核心修改：分级强制 English-only
# ─────────────────────────────────────────────
def get_system_prompt(k: int, strict_level: int = 0) -> str:
    """
    strict_level 0 → 正常提示
    strict_level 1 → 加重警告
    strict_level 2 → 极度严格，用于兜底重试
    """
    base = (
        "You are a computer vision expert. "
        "ALL output MUST be in ENGLISH. "
        "NEVER output Chinese, Japanese, Korean, or any non-English characters. "
        f"Output STRICTLY a JSON list of exactly {k} English strings. "
        "DO NOT include any explanation, preamble, or non-JSON text. "
        "Start your response with '[' and end with ']'. "
        f'Example: ["description one", "description two", "description three"]'
    )
    if strict_level == 1:
        base = (
            "CRITICAL INSTRUCTION: Respond in ENGLISH ONLY. "
            "DO NOT USE CHINESE CHARACTERS UNDER ANY CIRCUMSTANCES. "
        ) + base
    elif strict_level >= 2:
        base = (
            "!!! ENGLISH ONLY !!! DO NOT WRITE IN CHINESE !!! "
            "Every single word in your response must be English. "
            "If you write Chinese, your response is invalid. "
        ) + base
    return base

def build_prompt(node_type: str, concept: str, k: int, strict_level: int = 0) -> list:
    system = get_system_prompt(k, strict_level)
    clean_concept = clean_name(concept)

    # 在 user 侧也加入语言约束，避免模型忽略 system prompt
    lang_warning = (
        "IMPORTANT: Write ALL descriptions in ENGLISH only. "
        "Do NOT use Chinese or any other language. "
    )

    base_req = (
        f"{lang_warning}"
        f"Generate {k} DIFFERENT fine-grained English visual descriptions "
        f"for the physical concept '{clean_concept}'. "
        "Each description MUST explicitly mention: "
        "1. Texture, 2. Shape/Deformation, 3. Color/Light. "
        "Keep each description under 40 words. "
        "Output ONLY the JSON list, nothing else."
    )

    if node_type == "attr":
        user = (
            f"Describe how the attribute '{clean_concept}' visually alters a generic object. "
            + base_req
        )
    elif node_type == "obj":
        user = (
            f"Describe the typical standalone visual appearance of the object '{clean_concept}'. "
            + base_req
        )
    else:  # comp
        user = (
            f"Describe the combined visual features of '{clean_concept}', "
            "focusing on changes the attribute causes on the object. "
            + base_req
        )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

# ─────────────────────────────────────────────
# 中文检测 —— 宽松阈值：允许极少量误识别字符，但捕获真正的中文输出
# ─────────────────────────────────────────────
def contains_chinese(text: str) -> bool:
    """
    统计 CJK 字符占比。
    单个偶发字符（< 3 个）不触发拒绝，防止极少数 unicode 误判；
    但真正的中文句子必然远超阈值。
    """
    cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]', text)
    return len(cjk_chars) >= 3  # 3 个或以上 CJK 字符即判定为中文输出

# ─────────────────────────────────────────────
# JSON 提取
# ─────────────────────────────────────────────
def extract_json_list(text: str, expected_k: int) -> List[str]:
    # 1. 清理 Markdown 代码块标记后直接解析
    clean_text = re.sub(r'```(?:json)?', '', text).strip()
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, list) and len(parsed) >= expected_k:
            items = [str(i).strip() for i in parsed[:expected_k]]
            if not any(contains_chinese(i) for i in items):
                return items
    except Exception:
        pass

    # 2. 提取第一个完整的 JSON 数组
    array_match = re.search(r'\[.*?\]', clean_text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list) and len(parsed) >= expected_k:
                items = [str(i).strip() for i in parsed[:expected_k]]
                if not any(contains_chinese(i) for i in items):
                    return items
        except Exception:
            pass

    # 3. 正则抓取引号内的英文字符串（明确排除 CJK）
    pattern = r'["\']((?:[^"\'\\]|\\.)+)["\']'
    matches = re.findall(pattern, clean_text)
    candidates = [
        m.strip() for m in matches
        if len(m.strip()) > 8 and not contains_chinese(m)
    ]
    if len(candidates) >= expected_k:
        return candidates[:expected_k]

    # 4. 按行分割兜底
    lines = []
    for line in clean_text.split('\n'):
        line = line.strip().strip('"').strip("'").strip(',').strip('[').strip(']').strip()
        if len(line) > 10 and not contains_chinese(line):
            lines.append(line)
    if len(lines) >= expected_k:
        return lines[:expected_k]

    return []

# ─────────────────────────────────────────────
# 生成与校验 —— 核心修改：分级重试时升级 prompt 严格度
# ─────────────────────────────────────────────
def generate_candidates(
    tokenizer,
    model,
    node_type: str,
    concept: str,
    max_new_tokens: int,
    max_retries: int,
    expected_k: int,
    log_key: str,
) -> List[str]:
    """
    重试策略：
      attempt 0-1 → strict_level=0, temperature 从 0.3 开始
      attempt 2-3 → strict_level=1 (加重警告), temperature 适中
      attempt 4+  → strict_level=2 (极度严格) + 显式 few-shot 示例
    """
    last_response = ""

    for attempt in range(max_retries):
        # 根据重试轮次选择严格等级和温度
        if attempt < 2:
            strict_level = 0
            temperature  = 0.3 + attempt * 0.1
        elif attempt < 4:
            strict_level = 1
            temperature  = 0.5
        else:
            strict_level = 2
            temperature  = 0.6

        messages = build_prompt(node_type, concept, expected_k, strict_level)

        # 高严格等级时追加一个 assistant 前缀，强制模型以 '[' 开头
        # （利用 Qwen 的 prefix forcing 特性）
        if strict_level >= 1:
            messages.append({"role": "assistant", "content": "["})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(strict_level < 1)
        )
        # 若追加了 assistant 前缀，apply_chat_template 已处理，无需额外 add_generation_prompt
        if strict_level >= 1 and not text.rstrip().endswith("["):
            text = text.rstrip() + "["

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )

        out_ids  = [o[len(i):] for i, o in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        # 若使用了前缀强制，把 '[' 拼回去
        if strict_level >= 1:
            response = "[" + response

        last_response = response

        if contains_chinese(response):
            continue

        parsed_list = extract_json_list(response, expected_k)
        if parsed_list:
            return parsed_list

    print(f"\n[!] '{log_key}' 在 {max_retries} 次重试后仍生成失败。最后一次模型输出：")
    print(f"--- START OUTPUT ---\n{last_response}\n--- END OUTPUT ---")
    return [f"[FAILED] {log_key}"] * expected_k   # 使用有标记的占位符，方便后续批量重处理


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    attrs, objs, pairs = load_vocab(args.data_root)

    # ── 断点续跑 ──
    results = {}
    if os.path.exists(args.save_path):
        with open(args.save_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                print(f"[Resume] 发现已有进度，加载了 {len(results)} 条候选记录。")
            except json.JSONDecodeError:
                pass

    # ── 加载本地模型 ──
    print(f"\n[Model] 加载本地模型：{args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"[Model] 加载完成\n")

    def save_progress():
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_generated = 0

    def need_regenerate(key: str) -> bool:
        """判断一个 key 是否需要重新生成"""
        if key not in results:
            return True
        val = results[key]
        if not isinstance(val, list) or len(val) != args.k_candidates:
            return True
        # 重新生成含中文或失败占位符的条目
        for item in val:
            if contains_chinese(item) or item.startswith("[FAILED]"):
                return True
        return False

    def process_nodes(node_list, node_type, desc_name):
        nonlocal total_generated
        print(f"\n[Stage] 生成 {desc_name} 候选（共 {len(node_list)} 个）...")
        for item in tqdm(node_list, desc=desc_name):
            if isinstance(item, tuple):
                key          = f"{clean_name(item[0])} {clean_name(item[1])}"
                concept_name = key
            else:
                key          = clean_name(item)
                concept_name = key

            if not need_regenerate(key):
                continue

            candidates = generate_candidates(
                tokenizer, model,
                node_type, concept_name,
                args.max_new_tokens, args.max_retries, args.k_candidates,
                key,
            )

            results[key]     = candidates
            total_generated += 1

            if total_generated % 20 == 0:
                save_progress()

    if "attr" in args.node_types: process_nodes(attrs,  "attr", "属性节点(attr)")
    if "obj"  in args.node_types: process_nodes(objs,   "obj",  "物体节点(obj)")
    if "comp" in args.node_types: process_nodes(pairs,  "comp", "组合节点(comp)")

    save_progress()

    # ── 统计失败条目 ──
    failed = [k for k, v in results.items()
              if isinstance(v, list) and any(str(i).startswith("[FAILED]") for i in v)]
    print(f"\n[✓] 候选生成全部完成！")
    print(f"    本次新生成：{total_generated} 个")
    print(f"    文件总条数：{len(results)} 条")
    if failed:
        print(f"    ⚠ 仍有 {len(failed)} 条失败（含 [FAILED] 占位符）：{failed[:10]}")
    print(f"    保存路径：{args.save_path}")

if __name__ == "__main__":
    main()