"""Full-corpus E5 pilot with Qwen text-only control and CPU query timings."""
import argparse,sys,time,platform
from pathlib import Path
import numpy as np
import torch
from .core import read,write,digest,metrics
from .review_live_dtm_ranker import REQUESTS
from .review_consensus_live import REQUESTS as NEW_REQUESTS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/embeddings'))
from e5_small import E5Small,MANIFEST


def main(stage):
    src=ROOT/'output/balanced_benchmark_20261117';out=ROOT/'output/e5_small';data=src/'data'
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers]
    texts=[f'Title: {p["title"]}\nAbstract: {p["abstract"][:1800]}' for p in papers]
    meta=read(MANIFEST);signature=digest(dict(model=meta,texts=texts,paper_ids=ids,prefix='passage: ',max_length=512,pooling='masked_mean',dtype='float32'))
    if stage=='encode':
        device='cuda' if torch.cuda.is_available() else 'cpu';encoder=E5Small(device=device)
        cp=out/'documents_checkpoint.pt';start=0;chunks=[]
        if cp.exists():
            saved=torch.load(cp,weights_only=False,map_location='cpu');assert saved['signature']==signature
            chunks=[saved['vectors']];start=len(chunks[0])
        beginning=time.perf_counter()
        for i in range(start,len(texts),256):
            chunks.append(encoder.encode(texts[i:i+256],batch_size=32))
            if (i//256)%4==3 or i+256>=len(texts):
                temp=cp.with_suffix('.tmp');torch.save(dict(signature=signature,vectors=torch.cat(chunks)),temp);temp.replace(cp)
                print('Encoded',min(i+256,len(texts)),'/',len(texts),flush=True)
        vectors=torch.cat(chunks).numpy();assert vectors.shape==(len(ids),384) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        np.savez_compressed(out/'documents.npz',vectors=vectors)
        lengths=encoder.tokenizer(['passage: '+t for t in texts],truncation=False,add_special_tokens=True,return_length=True)['length']
        write(out/'documents_manifest.json',dict(signature=signature,paper_ids=ids,model=meta,input='Title+abstract[:1800]',max_tokens=512,truncated_documents=sum(n>512 for n in lengths),build_device=device,build_seconds_this_run=time.perf_counter()-beginning,vector_bytes=vectors.nbytes))
        return
    assert read(out/'documents_manifest.json')['signature']==signature
    queries=[q for q in read(data/'queries.json') if q['split']=='test'];js={j['query_id']:j for j in read(data/'judgments.json')}
    live=list(REQUESTS)+[(name+'_new',q) for name,q in NEW_REQUESTS]
    inputs=[q['query'] for q in queries]+[q for _,q in live]
    import psutil
    proc=psutil.Process();before=proc.memory_info().rss;begin=time.perf_counter();encoder=E5Small(device='cpu',threads=4);load_seconds=time.perf_counter()-begin;loaded_rss=proc.memory_info().rss
    encoder.encode(['논문 검색 준비'],query=True)
    with np.load(out/'documents.npz') as f:dv=f['vectors']
    vec=torch.from_numpy(dv);qvecs=[];timings=[]
    for text in inputs:
        start=time.perf_counter();qv=encoder.encode([text],query=True,batch_size=1)[0];encoded=time.perf_counter();scores=vec@qv;torch.topk(scores,3);finished=time.perf_counter()
        qvecs.append(qv);timings.append(dict(encode_ms=(encoded-start)*1000,search_ms=(finished-start)*1000))
    qvectors=torch.stack(qvecs).numpy();np.savez_compressed(out/'query_vectors.npz',vectors=qvectors)
    local={int(p['node_idx']):i for i,p in enumerate(papers)};rows=[]
    qwen_meta=read(src/'embeddings/text_manifest.json');assert qwen_meta['paper_ids']==ids
    with np.load(src/'embeddings/text.npz') as f:qwen_docs=f['vectors']
    qm=read(src/'embeddings/queries_manifest.json');assert qm['query_ids']==[q['query_id'] for q in queries]
    with np.load(src/'embeddings/queries.npz') as f:qwen_q=f['raw']
    for qi,q in enumerate(queries):
        j=js[q['query_id']];scores={'e5_text':dv@qvectors[qi],'qwen_text':qwen_docs@qwen_q[qi]}
        for protocol in ('pooled','pooled_without_anchor'):
            excluded=int(q['anchor_node_idx']) if protocol.endswith('without_anchor') else None
            candidates=[local[int(n)] for n in j['candidate_node_indices'] if int(n)!=excluded]
            labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels'] if int(v['node_idx'])!=excluded}
            for arm,s in scores.items():
                ranked=[ids[i] for i in sorted(candidates,key=lambda i:(-float(s[i]),ids[i]))]
                for k in (3,10):rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],protocol=protocol,arm=arm,k=k,**metrics(ranked,labels,k),precision=sum(labels.get(pid,0)>0 for pid in ranked[:k])/k,strong_precision=sum(labels.get(pid,0)==2 for pid in ranked[:k])/k))
    old={(r['query_id'],r['protocol'],r['k']):r for r in read(src/'natural_language/metrics.json') if r['arm']=='text'}
    for r in rows:
        if r['arm']=='qwen_text' and r['k']==10:assert abs(r['ndcg']-old[(r['query_id'],r['protocol'],10)]['ndcg'])<1e-7
    live_results=[]
    qwen_live=torch.load(src/'live_embedding_review/queries.pt',weights_only=False,map_location='cpu') if (src/'live_embedding_review/queries.pt').exists() else None
    if qwen_live is not None:assert qwen_live['texts']==[q for _,q in REQUESTS]
    for n,(name,query) in enumerate(live):
        s=dv@qvectors[len(queries)+n];top=np.argsort(-s,kind='stable')[:5]
        live_results.append(dict(name=name,query=query,results=[dict(rank=r+1,paper_id=ids[i],title=papers[i]['title'],abstract=papers[i]['abstract'],score=float(s[i])) for r,i in enumerate(top)]))
        if qwen_live is not None and n<len(REQUESTS):
            control=qwen_docs@qwen_live['vectors'][n].numpy();ix=np.argsort(-control,kind='stable')[:5]
            live_results[-1]['qwen_text_results']=[dict(rank=r+1,paper_id=ids[i],title=papers[i]['title'],abstract=papers[i]['abstract']) for r,i in enumerate(ix)]
    summary={}
    for protocol in ('pooled','pooled_without_anchor'):
        summary[protocol]={arm:{str(k):{m:float(np.mean([r[m] for r in rows if r['protocol']==protocol and r['arm']==arm and r['k']==k])) for m in ('ndcg','recall','mrr','precision','strong_precision')} for k in (3,10)} for arm in ('e5_text','qwen_text')}
    summary['paired_e5_minus_qwen']={}
    for k in (3,10):
        a={r['query_id']:r for r in rows if r['protocol']=='pooled_without_anchor' and r['arm']=='qwen_text' and r['k']==k};b={r['query_id']:r for r in rows if r['protocol']=='pooled_without_anchor' and r['arm']=='e5_text' and r['k']==k}
        ds=np.array([b[q]['ndcg']-a[q]['ndcg'] for q in a]);anchors=np.array([r['anchor'] for r in a.values()]);groups=np.unique(anchors);rng=np.random.default_rng(42)
        draws=[np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
        summary['paired_e5_minus_qwen'][f'ndcg@{k}']=dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),win_tie_loss=[int((ds>1e-10).sum()),int((abs(ds)<=1e-10).sum()),int((ds< -1e-10).sum())])
    parameters=sum(p.numel() for p in encoder.model.parameters());summary['cpu']=dict(threads=4,processor=platform.processor(),model_parameters=parameters,parameter_bytes=sum(p.numel()*p.element_size() for p in encoder.model.parameters()),vector_bytes=dv.nbytes,query_encode_ms_median=float(np.median([v['encode_ms'] for v in timings])),query_search_ms_median=float(np.median([v['search_ms'] for v in timings])),query_search_ms_p95=float(np.quantile([v['search_ms'] for v in timings],.95)),model_load_seconds=load_seconds,rss_before_model=before,rss_after_query_runs=proc.memory_info().rss,notes='CPU FP32, batch1, warm model; excludes disk loading and concurrent load. RSS is whole process including benchmark metadata, not model-only.')
    write(out/'metrics.json',rows);write(out/'live_results.json',live_results);write(out/'timings.json',timings);write(out/'summary.json',summary)
    summary['cpu']['rss_after_model_load']=loaded_rss
    write(out/'summary.json',summary)
    write(out/'evaluation_manifest.json',dict(query_hash=digest(inputs),document_signature=signature,query_prefix='query: ',passage_prefix='passage: ',query_device='cpu',rewrite=False,graph=False,filters=False,baseline_reproduction='passed',notes=['Same title+abstract[:1800] source as Qwen text-only; tokenizer truncation differs.','Existing fixed-pool LLM labels, original anchor excluded in secondary evaluation; not full-corpus gold.','Live queries reuse earlier topics and are qualitative checks, not untouched holdout.']))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['encode','evaluate'],required=True);main(p.parse_args().stage)
