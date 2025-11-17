# The following code is referenced from ProgramFC (Pan et al., 2023).
# (https://github.com/teacherpeterpan/ProgramFC/tree/main)

import argparse
import os
from doc_db import DocDB
import json
from tqdm import tqdm

def get_jsonline_corpus(db_path, save_path):
    db = DocDB(db_path)
    corpus = []

    existing_ids = set()
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["id"])

    doc_ids = db.get_doc_ids()
    for doc_id in tqdm(doc_ids):
        text = db.get_doc_text(doc_id)
        title = doc_id.split("_0")[0]
        if title in existing_ids:
            continue
        rec = {"id": title, "contents": text}
        corpus.append(rec)

    with open(save_path, "a") as f:
        for rec in corpus:
            f.write(json.dumps(rec) + "\n")

    print(f'Successfully processed {len(corpus)} wiki paragraphs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Path to save jsonline corpus')
    args = parser.parse_args()

    print(f'Creating jsonline corpus at {args.save_path}')
    get_jsonline_corpus(args.db_path, args.save_path)