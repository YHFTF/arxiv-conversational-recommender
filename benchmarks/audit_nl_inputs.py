"""Audit the exact legacy document construction against its token limit."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/embeddings'))
from local_embeddings import MODEL,REVISION

def main():
    master=json.loads((ROOT/'subdataset/arxiv_master_final.json').read_text(encoding='utf-8'))
    t=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,cache_dir=str(ROOT/'output/hf_cache'),local_files_only=True)
    counts={'papers':len(master),'text_over_512':0,'combined_fits_512':0,'dtm_fully_after_512':0,'dtm_partially_after_512':0}
    for p in master:
        prefix=f'Title: {p["title"]}\nAbstract: {p["abstract"][:1800]}'
        tags='; '.join(term for k in ('domain','task','method') for term in p['knowledge'][k])
        a=len(t(prefix)['input_ids']);b=len(t(prefix+'\nD/T/M: '+tags)['input_ids'])
        counts['text_over_512']+=a>512
        if b<=512: counts['combined_fits_512']+=1
        elif a>=512: counts['dtm_fully_after_512']+=1
        else: counts['dtm_partially_after_512']+=1
    counts['note']='Existing abstract[:1800] input with 512-token truncation; fully_after based on prefix token length.'
    (ROOT/'output/nl_benchmark_gpt6_luna_run1/input_truncation_audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    main()
