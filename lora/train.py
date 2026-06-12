import os
import numpy as np
import torch

from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer

from lora.preprocess_dataset import create_docred_hf_dataset
load_dotenv()
def load_peft_model(
        model_path: str,
        peft_config: LoraConfig = None
):
    model = AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float16)
    if not peft_config:
        peft_config = LoraConfig(
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            task_type="CAUSAL_LM",
            lora_dropout = 0.05,
        )
    return get_peft_model(model, peft_config)







def train():
    data_dir = os.getenv('LORA_TRAIN_DATA_DIR', './data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    train_data_file = os.path.join(data_dir, 'train.json')
    rel_file = os.path.join(data_dir, 'relation_type.json')
    val_data_file = os.path.join(data_dir, 'val.json')
    model_path = os.getenv('LLM_MODEL_PATH', None)
    if not model_path:
        raise ValueError('未指定LLM_MODEL_PATH')


    train_data_set = create_docred_hf_dataset(
        data_set_file=train_data_file,
        rel_mapping_file=rel_file,
        tokenizer_name_or_path=model_path,
        max_seq_len=int(os.getenv('MAX_SEQ_LEN', '1024'))
    )
    val_data_set = create_docred_hf_dataset(
        data_set_file=val_data_file,
        rel_mapping_file=rel_file,
        tokenizer_name_or_path=model_path,
        max_seq_len=int(os.getenv('MAX_SEQ_LEN', '1024'))
    )

    train_args = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=["tensorboard"],  # 关键：要求将指标数据汇报给 TensorBoard
        load_best_model_at_end=True,
        logging_strategy='steps',
        logging_steps=1,
        optim="paged_adamw_8bit",
        eval_accumulation_steps=1,
    )

    model = load_peft_model(
        model_path
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data_set,
        eval_dataset=val_data_set,
    )
    trainer.train()




if __name__ == "__main__":
    train()



