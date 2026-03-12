"""
train.py  —  RGIPCOL 主入口
"""
import argparse, json, os, random
from datetime import datetime

import numpy as np
import torch
import yaml

from data import get_dataloader
from data.data_utils import load_config, print_dataset_stats
from models import RGIPCOL
from trainer import Trainer, Evaluator
from trainer.feasibility import build_feasibility_calibrator
from utils.metrics import format_metrics
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description="RGIPCOL: Relational Graph-Injected Soft Prompting for CZSL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",           type=str, required=True)
    p.add_argument("--clip_model_path",  type=str, default=None)
    p.add_argument("--dataset_path",     type=str, default=None)
    p.add_argument("--save_path",        type=str, default=None)
    p.add_argument("--glove_path",       type=str, default=None,
                   help="GloVe 文件路径（双阶段过滤 Stage-1 必需）")
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--lr_prefix",        type=float, default=None)
    p.add_argument("--lr_rgcn",          type=float, default=None)
    p.add_argument("--batch_size",       type=int,   default=None)
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--num_workers",      type=int,   default=None)
    p.add_argument("--weight_decay",     type=float, default=None)
    p.add_argument("--rgcn_num_layers",  type=int,   default=None)
    p.add_argument("--rgcn_num_bases",   type=int,   default=None)
    p.add_argument("--rgcn_dropout",     type=float, default=None)
    p.add_argument("--prefix_length",    type=int,   default=None)
    p.add_argument("--lambda_prefix",    type=float, default=None)
    p.add_argument("--lambda_rgcn",      type=float, default=None)
    p.add_argument("--feasibility_threshold", type=float, default=None,
                   help="旧版官方预计算文件的过滤阈值")
    p.add_argument("--feasibility_theta1",    type=float, default=None,
                   help="双阶段 Stage-1 GloVe 粗筛阈值")
    p.add_argument("--feasibility_theta2",    type=float, default=None,
                   help="双阶段 Stage-2 LLM 精筛阈值")
    p.add_argument("--use_two_stage_filter",  action="store_true", default=None,
                   help="启用双阶段可行性过滤（覆盖 yaml 配置）")
    p.add_argument("--bias_step",        type=float, default=None)
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--eval_only",        action="store_true")
    p.add_argument("--checkpoint",       type=str,   default=None)
    return p.parse_args()


def merge_args_to_cfg(cfg, args):
    mapping = {
        "clip_model_path"       : args.clip_model_path,
        "data_dir"              : args.dataset_path,
        "glove_path"            : args.glove_path,
        "lr"                    : args.lr,
        "lr_prefix"             : args.lr_prefix,
        "lr_rgcn"               : args.lr_rgcn,
        "batch_size"            : args.batch_size,
        "epochs"                : args.epochs,
        "num_workers"           : args.num_workers,
        "weight_decay"          : args.weight_decay,
        "rgcn_num_layers"       : args.rgcn_num_layers,
        "rgcn_num_bases"        : args.rgcn_num_bases,
        "rgcn_dropout"          : args.rgcn_dropout,
        "prefix_length"         : args.prefix_length,
        "lambda_prefix"         : args.lambda_prefix,
        "lambda_rgcn"           : args.lambda_rgcn,
        "feasibility_threshold" : args.feasibility_threshold,
        "feasibility_theta1"    : args.feasibility_theta1,
        "feasibility_theta2"    : args.feasibility_theta2,
        "bias_step"             : args.bias_step,
        "seed"                  : args.seed,
    }
    for k, v in mapping.items():
        if v is not None:
            cfg[k] = v
    # bool flag 单独处理
    if args.use_two_stage_filter:
        cfg["use_two_stage_filter"] = True
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    args = parse_args()

    cfg    = load_config(args.config)
    cfg    = merge_args_to_cfg(cfg, args)
    device = torch.device(
        args.device if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    set_seed(cfg.get("seed", 42))

    # ── 实验目录 ──
    if args.save_path:
        exp_dir = args.save_path
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(
            cfg.get("experiment_dir", "./experiments"),
            f"{ts}_{cfg.get('dataset', 'czsl')}",
        )
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print("=" * 62)
    print(f"  RGIPCOL | dataset={cfg.get('dataset')} | seed={cfg.get('seed',42)}")
    print(f"  device          : {device}")
    print(f"  save_path       : {exp_dir}")
    print(f"  use_two_stage   : {cfg.get('use_two_stage_filter', False)}")
    if cfg.get("use_two_stage_filter", False):
        print(f"  glove_path      : {cfg.get('glove_path', 'N/A')}")
        print(f"  theta1 / theta2 : "
              f"{cfg.get('feasibility_theta1', 0.4)} / "
              f"{cfg.get('feasibility_theta2', 0.5)}")
        print(f"  llm_weights     : {cfg.get('llm_edge_weights_path', 'N/A')}")
    print(f"  lr={cfg.get('lr')}  bs={cfg.get('batch_size')}  "
          f"epochs={cfg.get('epochs')}")
    print("=" * 62)

    # ── 数据 ──
    print("[Main] 加载数据集...")
    train_loader, train_dataset = get_dataloader(cfg, "train")
    val_loader,   val_dataset   = get_dataloader(cfg, "val")
    test_loader,  test_dataset  = get_dataloader(cfg, "test")
    print_dataset_stats(train_dataset)

    # ── 模型 ──
    print("[Main] 构建模型...")
    model = RGIPCOL(cfg=cfg, dataset=train_dataset, device=device).to(device)

    if args.checkpoint:
        print(f"[Main] 加载检查点：{args.checkpoint}")
        load_checkpoint(model, args.checkpoint, device=device)

    # ── 训练 ──
    if not args.eval_only:
        trainer = Trainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, val_dataset=val_dataset,
            cfg=cfg, experiment_dir=exp_dir, device=device,
        )
        trainer.train()
        best_ckpt = os.path.join(exp_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            print("[Main] 加载最优模型进行测试...")
            load_checkpoint(model, best_ckpt, device=device)

    # ── 测试集评估 ──
    print("\n[Main] ===== 测试集评估 =====")
    evaluator = Evaluator(model=model, dataset=test_dataset,
                          cfg=cfg, device=device)

    print("\n[Main] --- 闭合世界 ---")
    closed = evaluator.evaluate_closed_world(test_loader)
    print(f"  {format_metrics(closed)}")

    print("\n[Main] --- 开放世界 ---")
    # ── 统一使用工厂函数构建校准器 ──
    feas_calibrator = build_feasibility_calibrator(cfg, test_dataset)
    opened = evaluator.evaluate_open_world(test_loader, feas_calibrator)
    print(f"  {format_metrics(opened)}")

    # ── 保存结果 ──
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"closed_world": closed, "open_world": opened,
                   "config": cfg}, f, indent=2, default=str)
    print(f"\n[Main] 结果已保存：{results_path}")
    print("=" * 62)
    print("  最终测试结果汇总")
    print(f"  闭合世界 : {format_metrics(closed)}")
    print(f"  开放世界 : {format_metrics(opened)}")
    print("=" * 62)


if __name__ == "__main__":
    main()