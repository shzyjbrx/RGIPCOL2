# test.py
import argparse
import os
import yaml
import torch

from data.dataset import get_dataloader
from models.rgipcol import RGIPCOL
from trainer.evaluator import Evaluator
from trainer.feasibility import build_feasibility_calibrator


def main():
    parser = argparse.ArgumentParser(description="Test RGIPCOL Model")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--open_world", action="store_true")
    # 可在命令行临时覆盖双阶段参数
    parser.add_argument("--use_two_stage_filter", action="store_true", default=None)
    parser.add_argument("--feasibility_theta1",   type=float, default=None)
    parser.add_argument("--feasibility_theta2",   type=float, default=None)
    parser.add_argument("--glove_path",           type=str,   default=None)
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["num_workers"] = 0

    # 命令行覆盖
    if args.use_two_stage_filter:
        cfg["use_two_stage_filter"] = True
    if args.feasibility_theta1 is not None:
        cfg["feasibility_theta1"] = args.feasibility_theta1
    if args.feasibility_theta2 is not None:
        cfg["feasibility_theta2"] = args.feasibility_theta2
    if args.glove_path is not None:
        cfg["glove_path"] = args.glove_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 计算设备: {device}")
    print(f"[*] 双阶段过滤: {cfg.get('use_two_stage_filter', False)}")

    # 2. 加载数据集
    print(f"[*] 加载 {args.split} 数据集...")
    test_loader, test_dataset = get_dataloader(cfg, args.split)

    # 3. 初始化模型
    print("[*] 初始化 RGIPCOL 模型...")
    model = RGIPCOL(cfg, test_dataset, device).to(device)

    # 4. 加载权重
    print(f"[*] 加载权重：{args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 5. 评估
    evaluator   = Evaluator(model, test_dataset, cfg, device)
    world_type  = "Open-World" if args.open_world else "Closed-World"
    print(f"[*] 开始 {world_type} 评估（{args.split} set）...")

    with torch.no_grad():
        if args.open_world:
            # ── 统一使用工厂函数构建校准器 ──
            calibrator = build_feasibility_calibrator(cfg, test_dataset)
            metrics    = evaluator.evaluate_open_world(
                test_loader, feasibility_calibrator=calibrator
            )
        else:
            metrics = evaluator.evaluate_closed_world(test_loader)

    # 6. 打印结果
    print("\n" + "=" * 50)
    print(f" {world_type} 评估结果（{args.split} set）")
    print("=" * 50)
    print(f"  Best Seen Accuracy (S)   : {metrics.get('S',   0.0):.4f}")
    print(f"  Best Unseen Accuracy (U) : {metrics.get('U',   0.0):.4f}")
    print(f"  Harmonic Mean (HM)       : {metrics.get('HM',  0.0):.4f}")
    print(f"  Area Under Curve (AUC)   : {metrics.get('AUC', 0.0):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()