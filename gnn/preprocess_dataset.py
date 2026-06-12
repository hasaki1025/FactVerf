import json
import os
import unicodedata
from typing import List, Dict, Tuple, Set, Any

import torch
from pyarrow.lib import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from gnn.graph import EntityGraph, preprocess_graph
from utils.logger import get_logger

logger = get_logger()


def get_rel_mapping_and_labels(data_path: str) -> Tuple[Dict[str, Set[Tuple[str, str]]], Dict[str, str]]:
    rel_mapping = {}
    label_mapping = {}

    with open(data_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, desc=f"Converting {os.path.basename(data_path)}"):
            data = json.loads(line)
            claim_id = data.get('id')
            evidence_lists = data.get('evidence', [])
            rel_mapping[claim_id] = set()
            label_mapping[claim_id] = data['label']
            for ev_group in evidence_lists:
                for annotation in ev_group:
                    if len(annotation) >= 4:
                        page_id = annotation[2]
                        sentence_id = annotation[3]
                        if page_id is not None and sentence_id is not None:
                            # 在字段值上做 NFC 规范化，确保能正确处理
                            # JSON \uXXXX 转义解码后可能为 NFD 形式的字符
                            page_id = unicodedata.normalize('NFC', page_id)
                            rel = (page_id, sentence_id)
                            rel_mapping[claim_id].add(rel)

        return rel_mapping, label_mapping


def load_graph_data(
        raw_data_file: str,
        claim_llm_rel_file: str,
        evidence_llm_rel_file: str,
):
    claim_evi_rel_mapping, label_mapping = get_rel_mapping_and_labels(raw_data_file)

    with open(claim_llm_rel_file, 'r', encoding='utf-8') as f:
        claim_llm_rel = [json.loads(line) for line in f.readlines()]

    with open(evidence_llm_rel_file, 'r', encoding='utf-8') as f:
        evidence_llm_rel_raw = [json.loads(line) for line in f.readlines()]
        # 构建 evidence 字典：缺少 extraction_result 字段的条目跳过并记录日志
        evidence_llm_rel = {}
        for item in evidence_llm_rel_raw:
            eid = (unicodedata.normalize('NFC', item['page_id']), item['sentence_id'])
            if 'extraction_result' not in item:
                logger.warning(
                    f'evidence page_id={item["page_id"]} sentence_id={item["sentence_id"]} '
                    f'缺少 extraction_result 字段，设置为空'
                )
                evidence_llm_rel[eid] = {
                    'rel': [],
                    'content': item['text'],
                }
                continue

            evidence_llm_rel[eid] = {
                'rel': item['extraction_result'],
                'content': item['text'],
            }

    return claim_llm_rel, label_mapping, claim_evi_rel_mapping, evidence_llm_rel


def prepare_graph_data(
        raw_data_file: str,
        evidence_llm_rel_file: str,
        claim_llm_rel_file: str,
) -> List[EntityGraph]:
    claim_llm_rel, label_mapping, claim_evi_rel_mapping, evidence_llm_rel = (
        load_graph_data(raw_data_file, claim_llm_rel_file, evidence_llm_rel_file))
    graphs = []
    for item in claim_llm_rel:

        try:
            claim_id = item['id']
            claim_content = item['text']
            label = label_mapping.get(claim_id, None)
            if label is None:
                logger.warning('未找到 claim_id: {}的标签'.format(claim_id))
                continue

            # claim 缺少 extraction_result 字段时跳过该条目
            if 'extraction_result' not in item:
                logger.warning(
                    f'[claim_id={claim_id}] 缺少 extraction_result 字段，设置为空'
                )
                raw_claim_rels = []
            else:
                raw_claim_rels = item['extraction_result']

            # 过滤掉 head/relation/tail 为 None 的三元组，防止 None 进入实体列表
            claim_rel_list: List[Tuple[str, str, str]] = [
                (rel['head'], rel['relation'], rel['tail'])
                for rel in raw_claim_rels
                if rel.get('head') is not None
                and rel.get('relation') is not None
                and rel.get('tail') is not None
            ]

            evidence_content: Dict[Tuple[str, str], str] = {}
            evidence_rel_list: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

            for evidence_id in claim_evi_rel_mapping.get(claim_id, []):
                try:
                    evidence_meta_data = evidence_llm_rel.get(
                        evidence_id,
                    )

                    if evidence_meta_data is None:
                        logger.warning(f'未找到 evidence_id: {evidence_id} 的具体内容， 所属claims id 为{claim_id}，跳过该 evidence')
                        continue

                    if evidence_id not in evidence_content:
                        evidence_content[evidence_id] = evidence_meta_data['content']

                    if evidence_id not in evidence_rel_list:
                        evidence_rel_list[evidence_id] = [
                            (rel['head'], rel['relation'], rel['tail'])
                            for rel in evidence_meta_data['rel']
                            if rel.get('head') is not None
                            and rel.get('relation') is not None
                            and rel.get('tail') is not None
                        ]
                except KeyError as e:
                    logger.error(
                        f'[claim_id={claim_id}] evidence_id={evidence_id} 缺少必要字段 {e}，跳过该 evidence',
                        exc_info=True,
                    )
                    continue

            graphs.append(
                preprocess_graph(
                    claim_content=claim_content,
                    label=label,
                    evidence_content=evidence_content,
                    claim_rel_list=claim_rel_list,
                    evidence_rel_list=evidence_rel_list,
                )
            )
        except KeyError as e:
            claim_id_hint = item.get('id', '<unknown>')
            logger.error(
                f'[claim_id={claim_id_hint}] 缺少必要字段 {e}，跳过该数据条目',
                exc_info=True,
            )
        except Exception as e:
            claim_id_hint = item.get('id', '<unknown>')
            logger.error(
                f'[claim_id={claim_id_hint}] 处理时发生未知错误: {e}，跳过该数据条目',
                exc_info=True,
            )


    return graphs



def batch_encode_graph_data(
    graph_list: List[EntityGraph],
    model,
    tokenizer,
    batch_size: int = 512,
    max_seq_length: int = 128,
)->List[Dict[str, Tensor]]:
    encode_node_list = [graph.node_list for graph in graph_list]
    encode_edge_list = [graph.edge_list for graph in graph_list]
    labels = torch.tensor([graph.label for graph in graph_list], dtype=torch.long)


    # ── 0. 确定运行设备并将模型移至对应设备 ─────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f'batch_encode_graph_data 使用设备: {device}')

    # ── 1. 展平所有节点/边文本，并记录每张图的偏移量 ──────────────────────────────
    all_node_texts: List[str] = []
    all_edge_texts: List[str] = []
    node_counts: List[int] = []
    edge_counts: List[int] = []

    for node_texts, edge_texts in zip(encode_node_list, encode_edge_list):
        node_counts.append(len(node_texts))
        edge_counts.append(len(edge_texts))
        all_node_texts.extend(node_texts)
        all_edge_texts.extend(edge_texts)

    def mean_pool_encode(texts: List[str], desc: str = 'Encoding') -> torch.Tensor:
        """对文本列表批量编码，返回 Tensor[len(texts), hidden_size]。
        编码结果为最后一层隐藏状态经 attention_mask 平均池化后的向量。
        空列表直接返回空 Tensor，hidden_size 取模型配置值。
        """
        if len(texts) == 0:
            hidden_size = model.config.hidden_size
            return torch.zeros(0, hidden_size)

        all_embeddings: List[torch.Tensor] = []
        model.eval()

        total_batches = (len(texts) + batch_size - 1) // batch_size
        with torch.no_grad():
            for start in tqdm(
                range(0, len(texts), batch_size),
                total=total_batches,
                desc=desc,
                unit='batch',
            ):
                batch_texts = texts[start: start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors='pt',
                )
                # 将输入移至目标设备
                encoded = {k: v.to(device) for k, v in encoded.items()}

                outputs = model(**encoded)
                last_hidden = outputs.last_hidden_state  # [B, seq_len, H]
                attention_mask = encoded['attention_mask']  # [B, seq_len]

                # 平均池化：对有效 token 的隐藏状态取均值
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                sum_hidden = (last_hidden * mask_expanded).sum(dim=1)  # [B, H]
                count = mask_expanded.sum(dim=1).clamp(min=1e-9)       # [B, 1]
                mean_hidden = sum_hidden / count                         # [B, H]

                all_embeddings.append(mean_hidden.cpu())

        return torch.cat(all_embeddings, dim=0)  # [N, H]

    # ── 2. 批量编码节点和边 ────────────────────────────────────────────────────────
    total_nodes = len(all_node_texts)
    total_edges = len(all_edge_texts)
    all_node_embeddings = mean_pool_encode(
        all_node_texts,
        desc=f'Encoding nodes ({total_nodes} texts)',
    )  # [total_nodes, H]
    all_edge_embeddings = mean_pool_encode(
        all_edge_texts,
        desc=f'Encoding edges ({total_edges} texts)',
    )  # [total_edges, H]

    # ── 3. 按图切分编码结果，组装返回值 ──────────────────────────────────────────
    encoded_graphs: List[Dict[str, torch.Tensor]] = []
    node_offset = 0
    edge_offset = 0

    for i, graph in enumerate(graph_list):
        n_nodes = node_counts[i]
        n_edges = edge_counts[i]

        encode_node = all_node_embeddings[node_offset: node_offset + n_nodes]  # [num_nodes, H]
        encode_edge = all_edge_embeddings[edge_offset: edge_offset + n_edges]  # [num_edges, H]

        # rel_indices: List[List[int]] -> Tensor[2, num_edges]（PyG 标准格式）
        if n_edges > 0:
            edge_indices = torch.tensor(graph.rel_indices, dtype=torch.long).t().contiguous()  # [2, E]
        else:
            edge_indices = torch.zeros(2, 0, dtype=torch.long)

        encoded_graphs.append({
            'encode_node': encode_node,    # Tensor[num_nodes, H]
            'encode_edge': encode_edge,    # Tensor[num_edges, H]
            'edge_indices': edge_indices,  # Tensor[2, num_edges]
            'label': labels[i],            # scalar Tensor
        })

        node_offset += n_nodes
        edge_offset += n_edges

    return encoded_graphs


class FEVERDataSet(torch.utils.data.Dataset):


    def __init__(
            self,
            raw_data_file: str,
            claim_llm_rel_file: str,
            evidence_llm_rel_file: str,
            embedding_batch_size: int = 512,
            embedding_max_seq_length: int = 128,
            embedding_model: str = 'microsoft/deberta-v3-base'
    ):
        # ── 本地缓存机制 ────────────────────────────────────────────────────────────
        # 缓存文件与 raw_data_file 放在同一目录，文件名编码了所有影响输出的关键参数，
        # 避免参数变化时误用旧缓存。
        _model_tag = embedding_model.replace('/', '_')
        _cache_name = (
            f"{os.path.splitext(os.path.basename(raw_data_file))[0]}"
            f"__{_model_tag}"
            f"__seqlen{embedding_max_seq_length}"
            f".pt"
        )
        _cache_dir = os.path.dirname(os.path.abspath(raw_data_file))
        _cache_path = os.path.join(_cache_dir, _cache_name)

        if os.path.exists(_cache_path):
            # 命中缓存：直接加载，跳过耗时的图构建和编码步骤
            logger.info(f"发现本地缓存，直接加载: {_cache_path}")
            self.data = torch.load(_cache_path, weights_only=False)
        else:
            # 未命中缓存：走完整数据处理流程
            logger.info(f"未发现本地缓存，开始数据处理流程...")
            self.graphs = prepare_graph_data(
                raw_data_file=raw_data_file,
                evidence_llm_rel_file=evidence_llm_rel_file,
                claim_llm_rel_file=claim_llm_rel_file,
            )
            self.embedding_model = AutoModel.from_pretrained(embedding_model, dtype="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)

            self.data = batch_encode_graph_data(
                self.graphs,
                self.embedding_model,
                self.tokenizer,
                embedding_batch_size,
                max_seq_length=embedding_max_seq_length,
            )

            # 将编码结果保存为 .pt 文件，供下次直接加载
            logger.info(f"数据处理完成，保存缓存至: {_cache_path}")
            torch.save(self.data, _cache_path)



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """

        :param idx: 索引
        :return: {
            'encode_node': Tensor[num_nodes, hidden_size],
            'encode_edge': Tensor[num_edges, hidden_size],
            'edge_indices': Tensor[2, num_edges],
            'label': scalar Tensor
        }
        """
        return self.data[idx]










