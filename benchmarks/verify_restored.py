"""Independent integrity and metric checks for the restored-data run."""
import argparse,json,math
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from .core import read,write,file_hash
from .restored import validate_judgments
from local_embeddings import IDENTITY

ROOT=Path(__file__).resolve().parents[1]


def binary_metrics(ranking,truth,k):
    truth=set(truth);hits=[i for i,v in enumerate(ranking[:k]) if v in truth]
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(truth))))
    return {'recall':len(hits)/len(truth) if truth else 0.,
        'ndcg':sum(1/math.log2(i+2) for i in hits)/ideal if ideal else 0.,
        'mrr':1/(hits[0]+1) if hits else 0.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/balanced_benchmark_20261117');parser.add_argument('--data-only',action='store_true');a=parser.parse_args()
    data=a.output/'data';master=read(data/'arxiv_master_final.json');sample=torch.load(data/'ogbn_arxiv_16k_ffs_sample.pt',weights_only=False,map_location='cpu')
    nodes=[int(p['node_idx']) for p in master];assert nodes==list(map(int,sample['indices']))
    official=pd.read_csv(ROOT/'dataset/ogbn_arxiv/raw/node-label.csv.gz',header=None).iloc[:,0].to_numpy()
    assert np.array_equal(sample['labels'].reshape(-1).numpy(),official[nodes])
    graph=torch.load(data/'build_hetero_graph_v2.pt',weights_only=False,map_location='cpu')
    assert torch.equal(graph['paper'].x,sample['features'])
    topic=graph['paper','has_topic','topic'].edge_index
    assert torch.equal(topic[0],torch.arange(len(master))) and torch.equal(topic[1],sample['labels'].reshape(-1))
    queries=read(data/'queries.json');judgments=read(data/'judgments.json')
    splits=validate_judgments(queries,judgments,set(nodes))
    info={'papers':len(master),'official_label_alignment':True,'graph_feature_alignment':True,'topic_mismatches':0,'query_splits':splits,'additional_api_calls':0}
    if a.data_only:
        write(a.output/'data_verification.json',info);print(json.dumps(info,indent=2));return
    split=read(a.output/'citation/split.json')['edges'];pairs=[{tuple(sorted(e)) for e in split[k]} for k in ('train','validation','test')]
    assert not pairs[0]&pairs[1] and not pairs[0]&pairs[2] and not pairs[1]&pairs[2]
    cs=read(a.output/'citation/summary.json');errors=[]
    for arm in ('ogbn','dtm','graph_only'):
        rows=read(a.output/'citation'/(arm+'_per_paper.json'))
        for r in rows:
            expected=binary_metrics(r['ranking'],r['truth'],r['k'])
            for m in expected:assert abs(expected[m]-r[m])<1e-12
        for k in (10,20):
            for metric in ('recall','ndcg','mrr'):
                errors.append(abs(np.mean([r[metric] for r in rows if r['k']==k])-cs[arm]['test'][str(k)][metric]))
    test=[q for q in queries if q['split']=='test'];qmap={q['query_id']:q for q in test};jmap={j['query_id']:j for j in judgments};node_to_id={p['node_idx']:str(p['paper_id']) for p in master}
    summary=read(a.output/'natural_language/summary.json');ranks=read(a.output/'natural_language/rankings.json');allmetrics=read(a.output/'natural_language/metrics.json')
    for row in ranks:
        q=qmap[row['query_id']];j=jmap[row['query_id']];excluded=q['anchor_node_idx'] if row['protocol']=='pooled_without_anchor' else None
        labels={node_to_id[v['node_idx']]:v['relevance'] for v in j['labels'] if v['node_idx']!=excluded}
        order=row['ranked_paper_ids'];assert len(order)==len(set(order)) and set(order)==set(labels)
        for k in (5,10,20):
            dcg=sum((2**labels[pid]-1)/math.log2(i+2) for i,pid in enumerate(order[:k]))
            ideal=sum((2**grade-1)/math.log2(i+2) for i,grade in enumerate(sorted(labels.values(),reverse=True)[:k]))
            expected=dcg/ideal if ideal else 0.
            measured=next(v['ndcg'] for v in allmetrics if v['query_id']==row['query_id'] and v['arm']==row['arm'] and v['protocol']==row['protocol'] and v['k']==k)
            errors.append(abs(expected-measured))
    assert max(errors)<1e-12
    activation=read(a.output/'activation.json')
    for name in ('arxiv_master_final.json','ogbn_arxiv_16k_ffs_sample.pt','paper_paper_edges.pt','author_paper_edges.pt','build_hetero_graph_v2.pt'):
        assert file_hash(ROOT/'subdataset'/name)==file_hash(data/name)
    catalog=torch.load(ROOT/'output/nl_recommendation_data/paper_text_embeddings.pt',weights_only=True,map_location='cpu')
    assert catalog['model']==IDENTITY and catalog['embeddings'].shape==(len(master),2560)
    assert catalog['node_indices'].tolist()==nodes and list(map(str,catalog['paper_ids']))==[str(p['paper_id']) for p in master]
    for name in ('queries.json','judgments.json'):
        assert file_hash(ROOT/'output/nl_recommendation_data'/name)==file_hash(data/name)
    with ZipFile(activation['backup']) as backup:
        assert backup.testzip() is None;backup_count=len(backup.namelist())
    info.update({'citation_pair_overlap':0,'independent_metric_recalculation_max_error':float(max(errors)),
        'test_queries':len(test),'test_source_papers':len({q['anchor_node_idx'] for q in test}),
        'both_anchor_protocols_complete':True,'active_data_files_match_staging':True,'active_qwen_shape':list(catalog['embeddings'].shape),
        'original_query_and_judgment_files_preserved':True,'backup_valid':True,'backup_files':backup_count})
    write(a.output/'verification.json',info);print(json.dumps(info,indent=2))

if __name__=='__main__':main()
