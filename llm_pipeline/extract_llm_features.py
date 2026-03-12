"""
extract_llm_features.py
-----------------------
将 LLM 生成的文本描述离线转换为 CLIP 特征向量，保存为 .pt 文件。
"""
import json
import torch
import clip
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./LLM/mit_node_descriptions.json")
    parser.add_argument("--save_path", type=str, default="./LLM/mit_node_features.pt")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    # 如果您有本地 CLIP 权重，可以替换为本地路径，例如 "./checkpoints/ViT-L-14.pt"
    parser.add_argument("--clip_path", type=str, default=None) 
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 CLIP 模型
    print(f"[*] 加载 CLIP 模型...")
    load_target = args.clip_path if args.clip_path else args.clip_model
    model, _ = clip.load(load_target, device=device, jit=False)
    model.eval()

    # 2. 读取 JSON 描述
    print(f"[*] 读取 JSON 描述文件：{args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        desc_dict = json.load(f)

    keys = list(desc_dict.keys())
    texts = list(desc_dict.values())
    features_dict = {}

    # 3. 分批提取特征
    print(f"[*] 开始提取特征，共 {len(texts)} 条...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[i : i + args.batch_size]
            batch_keys  = keys[i : i + args.batch_size]
            
            # Tokenize 并送入设备
            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            
            # 提取文本特征并进行 L2 归一化
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            
            # 转移回 CPU 以节省显存
            feats = feats.cpu()
            
            for k, feat in zip(batch_keys, feats):
                features_dict[k] = feat

    # 4. 保存为 .pt 文件
    torch.save(features_dict, args.save_path)
    print(f"[✓] 特征提取完成！已保存至：{args.save_path}")

if __name__ == "__main__":
    main()