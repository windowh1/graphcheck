cd datasets/ex-fever

# download ex-fever corpus
mkdir -p ./corpus
cd ./corpus
wget https://github.com/dependentsign/EX-FEVER/raw/refs/heads/master/data/wiki_db.db
wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db
cd ..

# build pyserini index
mkdir -p ./corpus/jsonl_corpus

# build jsonl corpus from wiki_db.db (gold evidence documents)
python build_jsonline_corpus_from_db.py \
    --db_path ./raw_data/wiki_db.db \
    --save_path ./corpus/jsonl_corpus/ex-fever_corpus.jsonl

# add additional documents from wiki_wo_links.db
python build_jsonline_corpus_from_db.py \
    --db_path ./raw_data/wiki_wo_links.db \
    --save_path ./corpus/jsonl_corpus/ex-fever_corpus.jsonl

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ./corpus/jsonl_corpus \
    --index ./corpus/index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 40 \
    --storePositions --storeDocvectors --storeRaw

rm ./corpus/wiki_db.db
rm ./corpus/wiki_wo_links.db
cd ../..