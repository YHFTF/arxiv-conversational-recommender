"""Actual full-corpus raw-query embedding retrieval, without benchmark labels."""
import os,sys
from pathlib import Path
import numpy as np
import torch
from .core import read,write,file_hash
from .review_live_dtm_ranker import REQUESTS
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/embeddings'),str(ROOT/'code/retrieval')]
from local_embeddings import LocalEmbedder,IDENTITY
from run_nl_recommendation import load_catalog


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch.set_num_threads(2)
    src=ROOT/'output/balanced_benchmark_20261117';out=src/'live_embedding_review';out.mkdir(exist_ok=True)
    texts=[q for _,q in REQUESTS];write(out/'requests.json',[dict(name=n,query=q) for n,q in REQUESTS])
    cache=out/'queries.pt'
    if cache.exists():
        saved=torch.load(cache,weights_only=False,map_location='cpu');assert saved['texts']==texts and saved['model']==IDENTITY;qv=saved['vectors']
    else:
        encoder=LocalEmbedder();qv=encoder.encode(texts,query=True);torch.save(dict(texts=texts,model=IDENTITY,vectors=qv),cache);del encoder;torch.cuda.empty_cache()
    papers,vectors,_=load_catalog(src/'data/arxiv_master_final.json',src/'embeddings/paper_text_embeddings.pt')
    results=[]
    for (name,query),v in zip(REQUESTS,qv):
        score=vectors@v;indices=torch.topk(score,5).indices.tolist()
        results.append(dict(name=name,query=query,results=[dict(rank=rank+1,paper_id=papers[i]['paper_id'],title=papers[i]['title'],abstract=papers[i]['abstract'],score=float(score[i])) for rank,i in enumerate(indices)]))
    write(out/'results.json',results)
    write(out/'manifest.json',dict(model=IDENTITY,corpus=16000,document_input='existing title+abstract+DTM embedding',query_input='raw unmodified query',llm_query_rewriting=False,graph_scoring=False,condition_reranking=False,benchmark_labels_read=False,requests='same 5 English/Korean pairs previously used for lightweight live review; not a new blind holdout',document_embedding_sha256=file_hash(src/'embeddings/paper_text_embeddings.pt')))
    for r in results:print(r['name'],[v['title'] for v in r['results'][:3]],flush=True)


if __name__=='__main__':main()
