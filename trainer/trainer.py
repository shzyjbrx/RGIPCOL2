"""
trainer.py
----------
RGIPCOL 训练主循环。

职责：
  - 构建优化器（前缀向量和 RGCN 可设不同学习率）
  - 训练循环（含梯度裁剪、日志记录）
  - 每个 epoch 后在 val 集评估，保存最优模型
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from trainer.evaluator import Evaluator


class Trainer:
    """
    Parameters
    ----------
    model      : RGIPCOL 实例
    train_loader : 训练 DataLoader
    val_loader   : 验证 DataLoader
    val_dataset  : 验证 CompositionDataset
    cfg          : 配置字典
    experiment_dir : 实验输出目录
    device       : 运行设备
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        val_dataset,
        cfg           : dict,
        experiment_dir: str,
        device        : torch.device,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.val_dataset    = val_dataset
        self.cfg            = cfg
        self.experiment_dir = experiment_dir
        self.device         = device

        # ── 优化器：只优化 prefix_vectors 和 RGCN 参数 ──
        param_groups = [
            {
                "params": list(model.prompt_learner.parameters()),
                "lr"    : cfg.get("lr_prefix", cfg.get("lr", 5e-5)),
                "name"  : "prefix",
            },
            {
                "params": list(model.rgcn.parameters()),
                "lr"    : cfg.get("lr_rgcn", cfg.get("lr", 1e-4)),
                "name"  : "rgcn",
            },
        ]
        self.optimizer = Adam(
            param_groups,
            weight_decay = cfg.get("weight_decay", 1e-5),
        )

        self.epochs = cfg.get("epochs", 20)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-7
        )

        # ── 日志 ──
        self.logger = Logger(
            log_path=os.path.join(experiment_dir, "train_log.txt"),
            print_to_console=True,
        )

        # ── 评估器 ──
        self.evaluator = Evaluator(
            model      = model,
            dataset    = val_dataset,
            cfg        = cfg,
            device     = device,
        )

        # ── 最优指标追踪 ──
        self.best_auc    = -1.0
        self.best_epoch  = -1

    # ────────────────────────────
    # 主训练循环
    # ────────────────────────────

    def train(self):
        self.logger.log(f"开始训练，共 {self.epochs} 个 epoch")
        self.logger.log(f"实验目录：{self.experiment_dir}")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # ── 训练一个 epoch ──
            train_loss = self._train_one_epoch(epoch)

            # ── 验证集评估 ──
            val_metrics = self.evaluator.evaluate_closed_world(
                self.val_loader
            )
            val_auc = val_metrics.get("AUC", 0.0)

            # ── 学习率调度 ──
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── 日志 ──
            elapsed = time.time() - epoch_start
            log_str = (
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss={train_loss:.4f} "
                f"Val_AUC={val_auc:.4f} "
                f"Val_HM={val_metrics.get('HM', 0.0):.4f} "
                f"Val_S={val_metrics.get('S', 0.0):.4f} "
                f"Val_U={val_metrics.get('U', 0.0):.4f} "
                f"LR={current_lr:.2e} "
                f"Time={elapsed:.1f}s"
            )
            self.logger.log(log_str)

            # ── 保存最优模型 ──
            if val_auc > self.best_auc:
                self.best_auc   = val_auc
                self.best_epoch = epoch
                save_checkpoint(
                    model     = self.model,
                    optimizer = self.optimizer,
                    epoch     = epoch,
                    metrics   = val_metrics,
                    path      = os.path.join(self.experiment_dir, "best_model.pt"),
                )
                self.logger.log(f"  ✓ 保存最优模型（epoch={epoch}, AUC={val_auc:.4f}）")

        self.logger.log(
            f"\n训练完成！最优 epoch={self.best_epoch}, 最优 Val_AUC={self.best_auc:.4f}"
        )

    # ────────────────────────────
    # 单 epoch 训练
    # ────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        # CLIP 部分始终保持 eval 模式（BN/Dropout 不影响冻结参数）
        self.model.clip.clip_model.eval()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        log_interval = self.cfg.get("log_interval", 50)

        for batch_idx, batch in enumerate(self.train_loader):
            images     = batch["image"].to(self.device)      # (B, 3, H, W)
            pair_idx   = batch["pair_idx"].to(self.device)   # (B,)

            # 过滤掉 pair_idx == -1 的样本（理论上训练集不应有）
            valid_mask = pair_idx >= 0
            if valid_mask.sum() == 0:
                continue
            images   = images[valid_mask]
            pair_idx = pair_idx[valid_mask]

            # ── 前向传播 + 损失 ──
            self.optimizer.zero_grad()
            loss = self.model(images, pair_idx)

            # ── 反向传播 ──
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.log(
                    f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"avg_loss={avg_loss:.4f}"
                )

        return total_loss / max(num_batches, 1)