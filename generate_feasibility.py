import os
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置路径 =================
# MIT-States 数据集所在的根目录
DATA_DIR = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/ut-zappos"
# 你的 GloVe 文件绝对路径
GLOVE_PATH = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/RGIPCOL2/glove/glove.6B.300d.txt"
# 输出的字典缓存文件路径
OUTPUT_PATH = "./glove/feasibility_ut-zappos_dict.pt"
# ============================================

def load_vocab(data_dir):
    """直接从 pairs 文本文件中动态提取所有属性、物体和可见的训练对"""
    split_dir = os.path.join(data_dir, "compositional-split-natural")
    
    attrs_set = set()
    objs_set = set()
    seen_pairs = []
    
    # 1. 读取 train_pairs.txt 获取 seen_pairs
    train_file = os.path.join(split_dir, "train_pairs.txt")
    print(f"[*] 正在读取训练集: {train_file}")
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                a, o = parts[0], parts[1]
                attrs_set.add(a)
                objs_set.add(o)
                seen_pairs.append((a, o))
                
    # 2. 读取测试集和验证集等其他文件，补全开放世界中才出现的 Unseen 属性和物体
    for fname in ["val_pairs.txt", "test_pairs.txt", "unseen_pairs.txt"]:
        filepath = os.path.join(split_dir, fname)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        attrs_set.add(parts[0])
                        objs_set.add(parts[1])
                        
    attrs = sorted(list(attrs_set))
    objs = sorted(list(objs_set))
    return attrs, objs, seen_pairs

def load_glove(glove_path, vocab):
    """加载 GloVe 词向量"""
    print(f"[*] 正在读取 GloVe 词向量: {glove_path} ...")
    embeddings = {}
    vocab_set = set(vocab)
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing GloVe"):
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab_set:
                embeddings[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    
    # MIT-States 中有少量合成词或罕见词，GloVe 中可能没有，使用零向量兜底
    for w in vocab:
        if w not in embeddings:
            print(f"[!] 警告：单词 '{w}' 不在 GloVe 词表中，将使用零向量代替。")
            embeddings[w] = np.zeros(300, dtype=np.float32)
            
    return embeddings

def cosine_sim(v1, v2):
    """计算余弦相似度"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    attrs, objs, seen_pairs = load_vocab(DATA_DIR)
    seen_set = set(seen_pairs)
    
    print(f"[*] 成功加载 {len(attrs)} 个属性, {len(objs)} 个物体, {len(seen_pairs)} 个可见组合")
    
    vocab = attrs + objs
    glove = load_glove(GLOVE_PATH, vocab)
    
    # 建立查找倒排索引
    applicable_attrs = {o: set() for o in objs}
    applicable_objs  = {a: set() for a in attrs}
    for a, o in seen_pairs:
        applicable_attrs[o].add(a)
        applicable_objs[a].add(o)
        
    feasibility_dict = {}
    total = len(attrs) * len(objs)
    
    print(f"[*] 开始计算 {total} 个组合的开放世界可行性分数...")
    with tqdm(total=total, desc="Computing Feasibility") as pbar:
        for a in attrs:
            vec_a = glove[a]
            for o in objs:
                if (a, o) in seen_set:
                    # 真实在训练集见过的组合，可行性直接拉满
                    feasibility_dict[(a, o)] = 1.0
                else:
                    vec_o = glove[o]
                    
                    # 1. attr feasibility (该物体见过的所有属性中，与当前属性最相似的得分)
                    attr_feas = 0.0
                    for a_seen in applicable_attrs.get(o, []):
                        sim = cosine_sim(vec_a, glove[a_seen])
                        if sim > attr_feas: attr_feas = sim
                        
                    # 2. obj feasibility (该属性见过的所有物体中，与当前物体最相似的得分)
                    obj_feas = 0.0
                    for o_seen in applicable_objs.get(a, []):
                        sim = cosine_sim(vec_o, glove[o_seen])
                        if sim > obj_feas: obj_feas = sim
                        
                    # 3. 综合得分
                    feasibility_dict[(a, o)] = (attr_feas + obj_feas) / 2.0
                
                pbar.update(1)
                
    print(f"[*] 计算完毕！")
    torch.save(feasibility_dict, OUTPUT_PATH)
    print(f"✅ 完美的字典格式可行性文件已生成: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()