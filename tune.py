"""
tune.py — Optuna 超参数自动调优脚本
====================================
调优目标：验证集 macro-F1（最大化）
调优超参数：
  - hidden_dim       : GNN 隐藏层维度  (categorical: [128, 256, 512])
  - num_layers       : MessagePassing 层数  (int: 1-4)
  - weight_decay     : AdamW 权重衰减  (float log: 1e-6 ~ 1e-2)
  - leaky_relu_slope : LeakyReLU 斜率  (float: 0.01 ~ 0.5)

用法示例：
  python tune.py \\
      --config config/gnn_config.yaml \\
      --dataset_dir dataset/FEVER \\
      --n_trials 30 \\
      --tune_epochs 10 \\
      --output_dir checkpoints

调优结束后，最优超参数会写入 config/gnn_config_tuned.yaml。
"""

import os
import copy
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnn.dataset import FEVERGraphDataset
from gnn.preprocess_dataset import FEVERDataSet
from gnn.model import LLMSKAN
from utils.early_stopping import EarlyStopping
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# 数据集加载（只加载一次，所有 Trial 共享）
# ---------------------------------------------------------------------------

def load_datasets(dataset_dir: str, batch_size: int):
    """加载 FEVER 训练集与验证集，返回两个 DataLoader。"""
    train_raw_path = os.path.join(dataset_dir, "train.jsonl")
    train_claim_path = os.path.join(dataset_dir, "train_claim.jsonl")
    train_evidence_path = os.path.join(dataset_dir, "train_evidence.jsonl")

    val_raw_path = os.path.join(dataset_dir, "shared_task_dev.jsonl")
    val_claim_path = os.path.join(dataset_dir, "val_claim.jsonl")
    val_evidence_path = os.path.join(dataset_dir, "val_evidence.jsonl")

    logger.info("Loading Training Data...")
    train_raw = FEVERDataSet(train_raw_path, train_claim_path, train_evidence_path)
    train_ds = FEVERGraphDataset(train_raw)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    logger.info("Loading Validation Data...")
    val_raw = FEVERDataSet(val_raw_path, val_claim_path, val_evidence_path)
    val_ds = FEVERGraphDataset(val_raw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 单 Trial 训练与评估
# ---------------------------------------------------------------------------

def run_trial(
    trial_params: dict,
    config_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_lr: float,
    tune_epochs: int,
    patience: int,
    device: torch.device,
    output_dir: str,
    trial_id: int,
) -> float:
    """
    使用给定超参数训练模型，返回最佳验证集 macro-F1。

    Args:
        trial_params: Optuna 采样到的超参数字典，包含：
            hidden_dim, num_layers, weight_decay, leaky_relu_slope
        config_path: yaml 配置路径（用于读取固定超参数）
        train_loader / val_loader: 数据加载器
        base_lr: 学习率（固定，不调优）
        tune_epochs: 每个 Trial 最大训练轮数
        patience: EarlyStopping 耐心值
        device: 训练设备
        output_dir: checkpoint 保存目录
        trial_id: Trial 编号（用于保存 checkpoint）

    Returns:
        该 Trial 在验证集上达到的最佳 macro-F1。
    """
    # 模型超参数 override（只覆盖被调优的字段）
    model_override = {
        "hidden_dim": trial_params["hidden_dim"],
        "num_layers": trial_params["num_layers"],
        "leaky_relu_slope": trial_params["leaky_relu_slope"],
    }

    model = LLMSKAN(config_path, config_override=model_override).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=trial_params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # 每个 Trial 保存独立 checkpoint，避免互相覆盖
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"trial_{trial_id}_best.pt")
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=ckpt_path)

    best_val_f1 = 0.0

    for epoch in range(1, tune_epochs + 1):
        # --- Train ---
        model.train()
        for batch in tqdm(train_loader, desc=f"Trial {trial_id} Epoch {epoch}/{tune_epochs} [Train]", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Trial {trial_id} Epoch {epoch}/{tune_epochs} [Val]", leave=False):
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        best_val_f1 = max(best_val_f1, val_f1)

        logger.info(
            f"[Trial {trial_id}] Epoch {epoch}/{tune_epochs} | "
            f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        # EarlyStopping 基于 val_loss
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"[Trial {trial_id}] Early stopping at epoch {epoch}.")
            break

    # 清理 Trial checkpoint，节省磁盘
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return best_val_f1


# ---------------------------------------------------------------------------
# Optuna 目标函数
# ---------------------------------------------------------------------------

def make_objective(
    config_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_lr: float,
    tune_epochs: int,
    patience: int,
    device: torch.device,
    output_dir: str,
):
    """工厂函数，返回闭包形式的 Optuna objective。"""

    def objective(trial: optuna.Trial) -> float:
        # ---- 超参数搜索空间 ----
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 768])
        num_layers = trial.suggest_int("num_layers", 1, 6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        leaky_relu_slope = trial.suggest_float("leaky_relu_slope", 0.01, 0.5)

        trial_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "weight_decay": weight_decay,
            "leaky_relu_slope": leaky_relu_slope,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"[Trial {trial.number}] 超参数: {trial_params}")
        logger.info(f"{'='*60}")

        val_f1 = run_trial(
            trial_params=trial_params,
            config_path=config_path,
            train_loader=train_loader,
            val_loader=val_loader,
            base_lr=base_lr,
            tune_epochs=tune_epochs,
            patience=patience,
            device=device,
            output_dir=output_dir,
            trial_id=trial.number,
        )

        logger.info(f"[Trial {trial.number}] 最终 Val F1: {val_f1:.4f}")
        return val_f1

    return objective


# ---------------------------------------------------------------------------
# 保存最优超参数到新的 yaml 配置
# ---------------------------------------------------------------------------

def save_best_config(
    base_config_path: str,
    best_params: dict,
    output_config_path: str,
):
    """
    将最优超参数合并到基础配置中，保存为新的 yaml 文件。

    Args:
        base_config_path: 原始 gnn_config.yaml 路径
        best_params: Optuna 找到的最优超参数字典
        output_config_path: 输出 yaml 路径（如 config/gnn_config_tuned.yaml）
    """
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新 model 字段
    config['model']['hidden_dim'] = best_params['hidden_dim']
    config['model']['num_layers'] = best_params['num_layers']
    config['model']['leaky_relu_slope'] = float(best_params['leaky_relu_slope'])

    # 更新 train 字段
    config['train']['weight_decay'] = float(best_params['weight_decay'])

    # 追加调优信息注释块（写在末尾）
    os.makedirs(os.path.dirname(output_config_path) or '.', exist_ok=True)
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        f.write(
            f"\n# === 由 tune.py (Optuna) 自动生成 ===\n"
            f"# 最优超参数:\n"
        )
        for k, v in best_params.items():
            f.write(f"#   {k}: {v}\n")

    logger.info(f"最优配置已保存至: {output_config_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna 超参数调优 — LLMSKAN GNN")
    parser.add_argument("--config", type=str, default="config/gnn_config.yaml",
                        help="基础超参数配置文件路径")
    parser.add_argument("--dataset_name", type=str, default="FEVER",
                        help="数据集名称（目前仅支持 FEVER）")
    parser.add_argument("--dataset_dir", type=str, default="dataset/FEVER",
                        help="数据集目录")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Optuna 调优的 Trial 数量（建议 20-50）")
    parser.add_argument("--tune_epochs", type=int, default=10,
                        help="每个 Trial 最大训练轮数（建议 5-15）")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tuning",
                        help="调优过程 checkpoint 保存目录")
    parser.add_argument("--output_config", type=str, default="config/gnn_config_tuned.yaml",
                        help="最优超参数输出 yaml 路径")
    parser.add_argument("--study_name", type=str, default="llmskan_hparam_search",
                        help="Optuna study 名称")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna study 持久化存储 URI（如 sqlite:///optuna.db）。"
                             "不指定则使用内存存储（不可恢复）。")
    args = parser.parse_args()

    if args.dataset_name.upper() != "FEVER":
        raise NotImplementedError(f"Dataset '{args.dataset_name}' is not supported yet.")

    # 加载基础配置
    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    train_cfg = base_cfg.get('train', {})

    batch_size = train_cfg.get('batch_size', 24)
    base_lr = train_cfg.get('learning_rate', 2e-4)
    patience = max(3, train_cfg.get('patience', 5) // 2)   # 调优时使用较短耐心值

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"Optuna Trials: {args.n_trials} | 每轮最大 Epoch: {args.tune_epochs}")

    # 加载数据集（全局一次，所有 Trial 共享）
    train_loader, val_loader = load_datasets(args.dataset_dir, batch_size)

    # 构建 Optuna study
    sampler = TPESampler(seed=42)  # 固定随机种子，保证可复现性
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",          # 最大化验证集 macro-F1
        sampler=sampler,
        storage=args.storage,
        load_if_exists=(args.storage is not None),
    )

    objective = make_objective(
        config_path=args.config,
        train_loader=train_loader,
        val_loader=val_loader,
        base_lr=base_lr,
        tune_epochs=args.tune_epochs,
        patience=patience,
        device=device,
        output_dir=args.output_dir,
    )

    logger.info("开始 Optuna 调优...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # 打印结果
    best_trial = study.best_trial
    logger.info("\n" + "=" * 60)
    logger.info("调优完成！")
    logger.info(f"最优 Trial ID : {best_trial.number}")
    logger.info(f"最优 Val F1   : {best_trial.value:.4f}")
    logger.info("最优超参数    :")
    for k, v in best_trial.params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # 保存最优配置
    save_best_config(
        base_config_path=args.config,
        best_params=best_trial.params,
        output_config_path=args.output_config,
    )

    logger.info(f"\n提示：使用最优配置进行正式训练：")
    logger.info(f"  python train.py --config {args.output_config}")


if __name__ == "__main__":
    main()
