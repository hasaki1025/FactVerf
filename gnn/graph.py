from typing import List, Dict, Tuple, Literal
from thefuzz import fuzz

def deduplicate_fuzzy(new_entity: str, existing_entities: List[str], threshold: int = 90) -> str:
    # 提前短路：如果是精确匹配，直接返回，节省 fuzz 计算时间
    if new_entity in existing_entities:
        return new_entity

    best_match = None
    highest_score = 0

    for exist_ent in existing_entities:
        score = fuzz.ratio(new_entity.lower(), exist_ent.lower())
        if score > highest_score:
            highest_score = score
            best_match = exist_ent

    if highest_score >= threshold:
        return best_match
    else:
        return new_entity


LABEL_MAPPING = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT ENOUGH INFO': 2
}
class EntityGraph:
    def __init__(self):
        self.node_list: List[str] = []
        self.edge_list: List[str] = []
        self.rel_indices: List[List[int]] = []
        # 使用 set 记录已添加的边，防止重复添加 (u_idx, rel_str, v_idx)
        self._seen_edges = set()
        self.label: int = None #  SUPPORTS |REFUTES|NOT ENOUGH INFO



    def set_label(self, label: str):
        if label not in LABEL_MAPPING:
            raise ValueError(f'The label {label} is not supported.')

        self.label = LABEL_MAPPING[label]

    def add_entity_node(self, entity_str: str, deduplicate: bool = True) -> int:
        """添加节点并返回其索引 ID。entity_str 为 None 或空字符串时直接返回 -1 并跳过添加。"""
        if not entity_str:  # 防止 None / 空字符串进入实体列表
            return -1

        if deduplicate:
            entity_str = deduplicate_fuzzy(entity_str, self.node_list)

        if entity_str in self.node_list:
            return self.node_list.index(entity_str)

        self.node_list.append(entity_str)
        return len(self.node_list) - 1

    def add_bidirectional_rel(self, head: str, relation: str, tail: str, deduplicate_nodes: bool = True):
        """添加双向无向边，并自动进行边去重。head/tail/relation 任意为 None 时跳过该边。"""
        if not head or not tail or not relation:  # 防止 None 字段导致后续崩溃
            return

        u_idx = self.add_entity_node(head, deduplicate=deduplicate_nodes)
        v_idx = self.add_entity_node(tail, deduplicate=deduplicate_nodes)

        if u_idx == -1 or v_idx == -1:  # 节点添加失败（None/空）则跳过该边
            return

        # 添加正向边
        edge_tuple_forward = (u_idx, relation, v_idx)
        if edge_tuple_forward not in self._seen_edges:
            self.edge_list.append(relation)
            self.rel_indices.append([u_idx, v_idx])
            self._seen_edges.add(edge_tuple_forward)

        # 添加反向边 (满足无向图设定)
        edge_tuple_backward = (v_idx, relation, u_idx)
        if edge_tuple_backward not in self._seen_edges:
            self.edge_list.append(relation)
            self.rel_indices.append([v_idx, u_idx])
            self._seen_edges.add(edge_tuple_backward)


def preprocess_graph(
        claim_content: str,
        label: Literal['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'],
        evidence_content: Dict[Tuple[str,str], str],  # 假设 key 是 evidence_id
        claim_rel_list: List[Tuple[str, str, str]],
        evidence_rel_list: Dict[Tuple[str,str], List[Tuple[str, str, str]]],
) -> EntityGraph:
    graph = EntityGraph()

    graph.set_label(label)

    # 1. 优先注册超长文本节点 (Claim 和 Evidence)，强制关闭模糊去重，提升性能
    graph.add_entity_node(claim_content, deduplicate=False)
    for ev_text in evidence_content.values():
        graph.add_entity_node(ev_text, deduplicate=False)

    # 2. 处理 Claim 中的实体与关系
    for claim_rel in claim_rel_list:
        # 修复 Bug: 解包获取完整三元组
        head_entity, relation, tail_entity = claim_rel[0], claim_rel[1], claim_rel[2]

        # 实体之间的关系
        graph.add_bidirectional_rel(head_entity, relation, tail_entity)
        # 实体归属于 Claim 的关系
        graph.add_bidirectional_rel(head_entity, 'belong to', claim_content)
        graph.add_bidirectional_rel(tail_entity, 'belong to', claim_content)

    # 3. 处理 Evidence 中的实体与关系
    for evidence_id, evidence_rels in evidence_rel_list.items():
        ev_text = evidence_content[evidence_id]

        for evidence_rel in evidence_rels:
            head_entity, relation, tail_entity = evidence_rel[0], evidence_rel[1], evidence_rel[2]

            # 实体之间的关系
            graph.add_bidirectional_rel(head_entity, relation, tail_entity)
            # 实体归属于当前 Evidence 的关系
            graph.add_bidirectional_rel(head_entity, 'belong to', ev_text)
            graph.add_bidirectional_rel(tail_entity, 'belong to', ev_text)

    return graph