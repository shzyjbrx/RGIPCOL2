"""
llm_pipeline/2_filter_with_clip.py
----------------------------------
第二步：加载候选描述文件，计算 CLIP 分数并筛选出最佳的细粒度描述。
"""

import os
import sys
import json
import torch
import clip
import argparse
from tqdm import tqdm

# 确保脚本可以引用主目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def select_best_description(clip_model, device, concept: str, candidates: list) -> str:
    # 只有一个候选，或者候选内容全都一样（比如触发了失败回退），直接返回
    if len(candidates) <= 1 or len(set(candidates)) == 1:
        return candidates[0]

    with torch.no_grad():
        # 1. 提取短名称(Anchor)的文本特征
        anchor_tokens = clip.tokenize([concept]).to(device)
        anchor_feat = clip_model.encode_text(anchor_tokens)
        anchor_feat = anchor_feat / anchor_feat.norm(dim=-1, keepdim=True)

        # 2. 提取所有长描述的文本特征
        desc_tokens = clip.tokenize(candidates, truncate=True).to(device)
        desc_feats = clip_model.encode_text(desc_tokens)
        desc_feats = desc_feats / desc_feats.norm(dim=-1, keepdim=True)

        # 3. 计算 Cosine Similarity
        similarities = (desc_feats @ anchor_feat.T).squeeze(1)
        best_idx = similarities.argmax().item()
        
    return candidates[best_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", type=str, default="./LLM/mit_candidate_descriptions.json")
    parser.add_argument("--save_path", type=str, default="./LLM/mit_filtered_descriptions.json")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--clip_path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] 读取候选描述文件: {args.candidate_path}")
    if not os.path.exists(args.candidate_path):
        raise FileNotFoundError(f"找不到文件 {args.candidate_path}，请先运行 1_generate_candidates.py")
        
    with open(args.candidate_path, "r", encoding="utf-8") as f:
        candidates_dict = json.load(f)

    print(f"[*] 加载 CLIP 模型用于质量筛选...")
    load_target = args.clip_path if args.clip_path else args.clip_model
    clip_model, _ = clip.load(load_target, device=device, jit=False)
    clip_model.eval()

    final_descriptions = {}

    print(f"[*] 开始进行 CLIP 相似度筛选...")
    for concept, candidates in tqdm(candidates_dict.items()):
        best_desc = select_best_description(clip_model, device, concept, candidates)
        final_descriptions[concept] = best_desc

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(final_descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n[✓] 筛选完毕！最终的高质量描述已保存至: {args.save_path}")

if __name__ == "__main__":
    main()