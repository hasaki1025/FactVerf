import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def plot_seq_len_distribution(seq_lengths, title_prefix="Sequence", save_path="seq_len_distribution.png"):
    """
    绘制 Token 长度分布直方图，并在图中标注关键分位数。
    """
    lengths_array = np.array(seq_lengths)

    p50 = np.percentile(lengths_array, 50)
    p90 = np.percentile(lengths_array, 90)
    p95 = np.percentile(lengths_array, 95)
    p99 = np.percentile(lengths_array, 99)
    mean_len = np.mean(lengths_array)
    max_len = np.max(lengths_array)

    plt.figure(figsize=(12, 6), dpi=120)

    n, bins, patches = plt.hist(
        lengths_array,
        bins=50,
        color='#4C72B0',
        edgecolor='black',
        alpha=0.75,
        label='Samples'
    )

    max_y = n.max()

    plt.axvline(p50, color='#E8A317', linestyle='--', linewidth=2, label=f'50% (Median): {p50:.0f}')
    plt.axvline(p90, color='#55A868', linestyle='--', linewidth=2, label=f'90%: {p90:.0f}')
    plt.axvline(p95, color='#C44E52', linestyle='-', linewidth=2, label=f'95%: {p95:.0f}')
    plt.axvline(p99, color='#8172B2', linestyle='--', linewidth=2, label=f'99%: {p99:.0f}')

    plt.title(f'{title_prefix} Token Length Distribution', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Token Length', fontsize=14)
    plt.ylabel('Frequency (Number of Samples)', fontsize=14)

    info_text = f"Total Samples: {len(lengths_array)}\nMax Length: {max_len:.0f}\nMean Length: {mean_len:.0f}"
    plt.text(0.95, 0.5, info_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 分布图已保存至: {save_path}")

def main():
    # 1. 加载数据
    print("Loading datasets...")
    with open('/home/lyq/projects/FactVerf/dataset/FEVER/train_claim.jsonl', 'r', encoding='utf-8') as f:
        train_claims_content = [json.loads(line)['text'] for line in f.readlines()]

    with open('/home/lyq/projects/FactVerf/dataset/FEVER/train_evidence.jsonl', 'r', encoding='utf-8') as f:
        train_evidences_content = [json.loads(line)['text'] for line in f.readlines()]

    # 2. 加载 tokenizer
    print("Loading tokenizer microsoft/deberta-v3-base...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # 3. 对文本进行编码并提取 input_ids，获取序列长度
    # 在这里按照您提到的基于 input_ids 获取序列长度的逻辑进行处理
    print("Calculating claim sequence lengths...")
    claims_encoded = tokenizer(train_claims_content, add_special_tokens=True)
    train_claims_seq_len = [len(seq) for seq in claims_encoded['input_ids']]

    print("Calculating evidence sequence lengths...")
    evidences_encoded = tokenizer(train_evidences_content, add_special_tokens=True)
    train_evidences_seq_len = [len(seq) for seq in evidences_encoded['input_ids']]

    # 4. 绘制并保存直方图
    plot_seq_len_distribution(
        train_claims_seq_len, 
        title_prefix="Train Claims", 
        save_path="/home/lyq/projects/FactVerf/dataset/FEVER/train_claims_seq_len.png"
    )
    
    plot_seq_len_distribution(
        train_evidences_seq_len, 
        title_prefix="Train Evidences", 
        save_path="/home/lyq/projects/FactVerf/dataset/FEVER/train_evidences_seq_len.png"
    )

if __name__ == '__main__':
    main()
