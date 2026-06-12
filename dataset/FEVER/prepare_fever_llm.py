import glob
import json
import os
import sqlite3
import unicodedata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def build_db(wiki_dir, db_path):
    print("Building SQLite DB for Wikipedia pages... This may take a few minutes if running for the first time.")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS lines (id TEXT, sentence_id INTEGER, text TEXT)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_id_sentence ON lines(id, sentence_id)')
    
    c.execute("SELECT count(*) FROM lines")
    if c.fetchone()[0] > 0:
        print("DB already populated. Skipping DB creation.")
        return conn
        
    files = glob.glob(os.path.join(wiki_dir, "*.jsonl"))
    if not files:
        print(f"Warning: No wiki-*.jsonl files found in {wiki_dir}")
        return conn
        
    for fpath in tqdm(files, desc="Processing wiki pages"):
        records = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                page_id = data.get('id', '')
                if not page_id:
                    continue
                lines_str = data.get('lines', '')
                for line_str in lines_str.split('\n'):
                    parts = line_str.split('\t')
                    if len(parts) >= 2:
                        try:
                            sentence_id = int(parts[0])
                            text = parts[1]
                            records.append((page_id, sentence_id, text))
                        except ValueError:
                            pass
        
        c.executemany('INSERT INTO lines VALUES (?, ?, ?)', records)
        conn.commit()
        
    print("Finished building DB.")
    return conn

def get_evidence_text(conn, page_id, sentence_id):
    c = conn.cursor()
    c.execute("SELECT text FROM lines WHERE id=? AND sentence_id=?", (page_id, sentence_id))
    row = c.fetchone()
    if row:
        return row[0]
    return ""





def process_dataset(input_path, evidence_out_path, claims_out_path, conn):
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist, skipping.")
        return
        
    print(f"Processing dataset: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(evidence_out_path, 'w', encoding='utf-8') as f_evident_out, \
         open(claims_out_path, 'w', encoding='utf-8') as f_claims_out:

        evidence_id_list = set()
        claims_list = []


        for line in tqdm(fin, desc=f"Converting {os.path.basename(input_path)}"):
            data = json.loads(line)
            claim_id = data.get('id')
            claim = data.get('claim')
            evidence_lists = data.get('evidence', [])
            claims_list.append({
                'id': claim_id,
                'text': claim,
            })

            for ev_group in evidence_lists:
                for annotation in ev_group:
                    if len(annotation) >= 4:
                        page_id = annotation[2]
                        sentence_id = annotation[3]
                        if page_id is not None and sentence_id is not None:
                            # NFC 规范化：train.jsonl 中 page_id 可能为 NFD 形式，
                            # 而 build_db 写入 SQLite 时 json.loads 解码后为 NFC。
                            # SQLite TEXT 比较是字节级别，NFD != NFC 会导致查不到数据。
                            page_id = unicodedata.normalize('NFC', page_id)
                            evidence_id_list.add((page_id, sentence_id))

        evidence_list = []
        for evidence_page_id, evidence_sentence_id in tqdm(evidence_id_list, desc="Retrieving evidence text"):
            evidence_text = get_evidence_text(conn, evidence_page_id, evidence_sentence_id)
            if evidence_text and len(evidence_text.strip()) > 0:
                evidence_list.append({
                    'page_id': evidence_page_id,
                    'sentence_id': evidence_sentence_id,
                    'text': evidence_text,
                })

        for claim in claims_list:
            f_claims_out.write(json.dumps(claim, ensure_ascii=False) + '\n')

        for evidence in evidence_list:
            f_evident_out.write(json.dumps(evidence, ensure_ascii=False) + '\n')



def main():
    base_dir = "./"
    wiki_dir = os.path.join(base_dir, "wiki-pages")
    db_path = os.path.join(base_dir, "wiki.db")
    
    conn = build_db(wiki_dir, db_path)
    
    # train_in = os.path.join(base_dir, "train.jsonl")
    # train_claim_out = os.path.join(base_dir, "train_claim.jsonl")
    # train_evidence_out = os.path.join(base_dir, "train_evidence.jsonl")
    # process_dataset(train_in,train_evidence_out, train_claim_out, conn)
    #
    # dev_in = os.path.join(base_dir, "shared_task_dev.jsonl")
    # dev_claim_out = os.path.join(base_dir, "val_claim.jsonl")
    # dev_evidence_out = os.path.join(base_dir, "val_evidence.jsonl")
    # process_dataset(dev_in, dev_evidence_out, dev_claim_out, conn)


    test_in = os.path.join(base_dir, "shared_task_test.jsonl")
    test_claim_out = os.path.join(base_dir, "test_claim.jsonl")
    test_evidence_out = os.path.join(base_dir, "test_evidence.jsonl")
    process_dataset(test_in, test_evidence_out, test_claim_out, conn)
    
    conn.close()

if __name__ == "__main__":
    main()
