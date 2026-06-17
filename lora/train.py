"""
通用 LoRA 微调入口脚本。

所有超参数由外部 YAML 配置文件统一控制；数据集通过 --dataset_name 选择加载器，
通过 --dataset_dir 指定数据根目录（可在命令行覆盖 config 中的默认值）。

支持的数据集名称（--dataset_name）：
  remote   —— Remote 多模态实体关系数据集（Qwen-VL，含图像）
  docred   —— DOCRED-FE 文本实体关系数据集（纯文本 LLM）

使用示例：
  # Remote 数据集
  python -m lora.train \\
    --config config/lora_remote.yaml \\
    --dataset_name remote \\
    --dataset_dir /media/shared_e/lyq/DataSet/REMOTE/datasets

  # DOCRED 数据集
  python -m lora.train \\
    --config config/lora_docred.yaml \\
    --dataset_name docred \\
    --dataset_dir /path/to/docred
"""

import argparse
import os
import sys

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from utils.logger import get_logger

logger = get_logger()

# ──────────────────────────────────────────────────────────────
# 1. 数据集注册表
# ──────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, callable] = {}


def register_dataset(name: str):
    """装饰器：将数据集加载函数注册到全局注册表。"""
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


@register_dataset("remote")
def load_remote_dataset(split: str, dataset_dir: str, config: dict):
    """
    加载 Remote 多模态实体关系数据集。
    config["dataset"] 中需包含：
      - train_file / val_file：JSON 文件名（相对 dataset_dir）
      - image_dir：图像目录名（相对 dataset_dir）
      - max_seq_len、num_proc
      - train_cache_dir / val_cache_dir（可选，相对 dataset_dir）
    config["model"]["model_path"]：Processor 所在路径
    """
    from lora.remote_lora.dataset import create_remote_dataset

    ds_cfg = config["dataset"]
    if split == "train":
        file_key, cache_key = "train_file", "train_cache_dir"
    elif split == "val":
        file_key, cache_key = "val_file", "val_cache_dir"
    else:  # "test"
        file_key, cache_key = "test_file", "test_cache_dir"

    dataset_file = os.path.join(dataset_dir, ds_cfg[file_key])
    image_dir = os.path.join(dataset_dir, ds_cfg.get("image_dir", ""))

    raw_cache = ds_cfg.get(cache_key, "")
    cache_dir = os.path.join(dataset_dir, raw_cache) if raw_cache else None

    return create_remote_dataset(
        dataset_file=dataset_file,
        image_dir=image_dir,
        tokenizer_name_or_path=config["model"]["model_path"],
        max_seq_len=ds_cfg.get("max_seq_len", 8192),
        num_proc=ds_cfg.get("num_proc", 16),
        cache_dir=cache_dir,
    )


@register_dataset("docred")
def load_docred_dataset(split: str, dataset_dir: str, config: dict):
    """
    加载 DOCRED-FE 文本实体关系数据集。
    config["dataset"] 中需包含：
      - train_file / val_file：JSON 文件名（相对 dataset_dir）
      - rel_mapping_file（可选）
      - max_seq_len、num_proc
    config["model"]["model_path"]：Tokenizer 所在路径
    """
    from lora.DOCRED_FE_LORA.preprocess_dataset import create_docred_hf_dataset
    ds_cfg = config["dataset"]
    if split == "train":
        file_key = "train_file"
    elif split == "val":
        file_key = "val_file"
    else:  # "test"
        file_key = "test_file"

    dataset_file = os.path.join(dataset_dir, ds_cfg[file_key])
    rel_file = ds_cfg.get("rel_mapping_file", "")
    rel_mapping_file = os.path.join(dataset_dir, rel_file) if rel_file else None

    return create_docred_hf_dataset(
        data_set_file=dataset_file,
        tokenizer_name_or_path=config["model"]["model_path"],
        max_seq_len=ds_cfg.get("max_seq_len", 2048),
        rel_mapping_file=rel_mapping_file,
        num_proc=ds_cfg.get("num_proc", 8),
    )


# ──────────────────────────────────────────────────────────────
# 2. 配置加载
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置字典。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"已从 {config_path} 加载配置")
    return cfg


# ──────────────────────────────────────────────────────────────
# 3. 模型加载
# ──────────────────────────────────────────────────────────────

def load_peft_model(config: dict):
    """根据 config["model"] 和 config["lora"] 加载 LoRA 包装后的模型。"""
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    # dtype 解析
    _dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    dtype = _dtype_map.get(model_cfg.get("torch_dtype", "float16"), torch.float16)

    logger.info(f"正在加载基座模型: {model_cfg['model_path']}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_path"],
        torch_dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────
# 4. TrainingArguments 构建
# ──────────────────────────────────────────────────────────────

def build_training_args(config: dict) -> TrainingArguments:
    """将 config["training"] 转换为 TrainingArguments 对象。"""
    t = config["training"]

    # 确保 output_dir 存在
    os.makedirs(t.get("output_dir", "checkpoints"), exist_ok=True)

    return TrainingArguments(
        output_dir=t.get("output_dir", "checkpoints"),
        max_steps=t.get("max_steps", -1),               # -1 = 由 num_train_epochs 决定
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=float(t.get("learning_rate", 2e-5)),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        optim=t.get("optim", "paged_adamw_8bit"),
        eval_strategy=t.get("eval_strategy", "epoch"),
        eval_steps=t.get("eval_steps", None),
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        logging_strategy=t.get("logging_strategy", "steps"),
        logging_steps=t.get("logging_steps", 10),
        report_to=t.get("report_to", ["tensorboard"]),
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", False),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        eval_accumulation_steps=t.get("eval_accumulation_steps", 1),
        remove_unused_columns=t.get("remove_unused_columns", False),
        use_cpu=t.get("use_cpu", False),
    )


# ──────────────────────────────────────────────────────────────
# 5. 主入口
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通用 LoRA 微调脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        metavar="PATH",
        help="YAML 配置文件路径，例如 config/lora_remote.yaml",
    )
    parser.add_argument(
        "--dataset_name", "-d",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"数据集名称，支持：{list(DATASET_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        metavar="DIR",
        help="数据集根目录，会覆盖配置文件中的相对路径基准",
    )
    return parser.parse_args()


def preprocess_logits_for_metrics(logits, labels):
    """
    在验证集/测试集进行 evaluation 时，HuggingFace 默认会将所有 batch 的 logits 累积到内存中。
    对于 LLM，logits 大小为 [batch_size, seq_len, vocab_size]，非常容易导致 OOM。
    该函数在每个 batch 后立即对 logits 进行 argmax 操作，将其压缩为 [batch_size, seq_len]。
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class SaveBeforeEvalCallback(TrainerCallback):
    """
    通过回调在评估前和训练结束后主动保存权重。
    """
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        # 如果即将进行评估且恰好当前步需要保存（例如 epoch 结束）
        # 我们提前调用 _save_checkpoint 保存，防止验证集 OOM 导致当前 epoch 成果丢失
        if control.should_evaluate and control.should_save:
            logger.info("【Callback】检测到即将进行评估，为了防止评估引发OOM导致白跑，提前进行模型保存...")
            self.trainer._save_checkpoint(self.trainer.model, trial=None, metrics=None)
            control.should_save = False  # 防止在主循环中重复保存

    def on_train_end(self, args, state, control, **kwargs):
        # 满足用户“在训练完成后保存权重”的要求（也可在 train() 中直接调用）
        final_dir = os.path.join(args.output_dir, "final")
        logger.info(f"【Callback】训练完成，正在保存最终模型到 {final_dir}")
        self.trainer.save_model(final_dir)


def train():
    args = parse_args()

    # ── 加载配置 ──
    config = load_config(args.config)
    logger.info(f"dataset_name={args.dataset_name}  dataset_dir={args.dataset_dir}")

    # ── 校验数据集 ──
    if args.dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {args.dataset_name}，"
            f"支持的名称: {list(DATASET_REGISTRY.keys())}"
        )
    load_fn = DATASET_REGISTRY[args.dataset_name]

    # ── 加载数据 ──
    logger.info("═══ 加载训练集 ═══")
    train_dataset = load_fn("train", args.dataset_dir, config)

    val_file_key = "val_file"
    has_val = bool(config.get("dataset", {}).get(val_file_key))
    val_dataset = None
    if has_val:
        logger.info("═══ 加载验证集 ═══")
        val_dataset = load_fn("val", args.dataset_dir, config)

    # ── 加载模型 ──
    logger.info("═══ 加载模型 ═══")
    model = load_peft_model(config)

    # ── 构建 TrainingArguments ──
    training_args = build_training_args(config)

    # ── 构建 Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.add_callback(SaveBeforeEvalCallback(trainer))

    # ── 开始训练 ──
    logger.info("═══ 开始训练 ═══")
    trainer.train()

    # 训练完成后的最终模型保存已集成到 SaveBeforeEvalCallback.on_train_end 中

    # ── 在测试集上评估 ──
    test_file_key = "test_file"
    has_test = bool(config.get("dataset", {}).get(test_file_key))
    if has_test:
        logger.info("═══ 加载测试集 ═══")
        test_dataset = load_fn("test", args.dataset_dir, config)
        logger.info("═══ 开始测试集评估 ═══")
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        logger.info(f"测试集评估结果: {test_results}")


if __name__ == "__main__":
    # 确保项目根目录在 PYTHONPATH 中（直接运行时需要）
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    train()
