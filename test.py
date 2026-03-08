# test.py
import argparse
import os
import yaml
import torch

from data.dataset import get_dataloader
from models.rgipcol import RGIPCOL
from trainer.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Test RGIPCOL2 Model")
    parser.add_argument("--config", type=str, required=True, help="配置文件的路径 (例如 configs/mit_states_config.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="保存的模型权重路径 (例如 saved_models/.../best_model.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="评估的数据集划分 (val 或 test)")
    parser.add_argument("--open_world", action="store_true", help="是否进行开放世界 (Open-World) 评估")
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用计算设备: {device}")

    # 2. 加载数据集
    print(f"[*] 正在加载 {args.split} 数据集...")
    # get_dataloader 预期返回 (dataloader, dataset_obj)
    test_loader, test_dataset = get_dataloader(cfg, args.split)

    # 3. 初始化模型结构
    print("[*] 正在初始化 RGIPCOL 模型结构...")
    model = RGIPCOL(cfg, test_dataset, device).to(device)

    # 4. 加载模型权重 (Checkpoint)
    print(f"[*] 正在从 {args.checkpoint} 加载权重...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容常见的 checkpoint 字典格式保存，加入 strict=False
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    # 必须将模型置于评估模式，关闭 Dropout 等
    model.eval()

    # 5. 初始化 Evaluator
    evaluator = Evaluator(model, test_dataset, cfg, device)

    # 6. 开始评估
    world_type = "Open-World" if args.open_world else "Closed-World"
    print(f"[*] 开始在 {args.split} 集上进行 {world_type} 评估...")
    
    with torch.no_grad():
        if args.open_world:
            # 开放世界评估 (目前暂未传入 feasibility_calibrator，进行最严苛的全空间匹配)
            metrics = evaluator.evaluate_open_world(test_loader, feasibility_calibrator=None)
        else:
            # 闭合世界评估
            metrics = evaluator.evaluate_closed_world(test_loader)

    # 7. 打印最终指标
    print("\n" + "="*50)
    print(f" {world_type} 评估结果 ({args.split} set):")
    print("="*50)
    print(f"  Best Seen Accuracy (S)   : {metrics.get('S', 0.0):.4f}")
    print(f"  Best Unseen Accuracy (U) : {metrics.get('U', 0.0):.4f}")
    print(f"  Harmonic Mean (HM)       : {metrics.get('HM', 0.0):.4f}")
    print(f"  Area Under Curve (AUC)   : {metrics.get('AUC', 0.0):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()