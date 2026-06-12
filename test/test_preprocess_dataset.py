from gnn.preprocess_dataset import prepare_graph_data, FEVERDataSet


def test_prepare_graph_data():
    raw_data_file = 'dataset/FEVER/train.jsonl'
    claim_llm_rel_file = 'dataset/FEVER/train_claim.jsonl'
    evidence_llm_rel_file = 'dataset/FEVER/train_evidence.jsonl'

    graphs = prepare_graph_data(
        raw_data_file, evidence_llm_rel_file, claim_llm_rel_file
    )
    assert len(graphs) >= 145449



def test_get_dataset():
    raw_data_file = 'dataset/FEVER/train.jsonl'
    claim_llm_rel_file = 'dataset/FEVER/train_claim.jsonl'
    evidence_llm_rel_file = 'dataset/FEVER/train_evidence.jsonl'

    dataset = FEVERDataSet(
        raw_data_file,
        claim_llm_rel_file,
        evidence_llm_rel_file,
    )
    assert len(dataset) == 145449


