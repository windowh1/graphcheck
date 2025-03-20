cd datasets/ex-fever

# download corpus for feverous
mkdir -p ./corpus
cd ./corpus
wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db
cd ..

mkdir -p ./corpus/jsonl_corpus

# build pyserini index from sqlite database
python build_jsonline_corpus_from_db.py \
    --db_path ./corpus/wiki_wo_links.db \
    --save_path ./corpus/jsonl_corpus/ex-fever_corpus.jsonl

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ./corpus/jsonl_corpus \
    --index ./corpus/index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 40 \
    --storePositions --storeDocvectors --storeRaw

rm ./corpus/wiki_wo_links.db