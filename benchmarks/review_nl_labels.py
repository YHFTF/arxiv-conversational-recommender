"""Inspect blinded 0/1/2 labels and ranked results without a server."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/nl_benchmark_gpt6_luna_run1'

def main():
    queries=json.loads((OUT/'queries.json').read_text(encoding='utf-8'))
    ranks=json.loads((OUT/'rankings.json').read_text(encoding='utf-8'))
    qrels=json.loads((OUT/'qrels.json').read_text(encoding='utf-8'))
    master=json.loads((ROOT/'subdataset/arxiv_master_final.json').read_text(encoding='utf-8'))
    needed={pid for r in ranks for pid in r['pool']}
    papers={str(p['paper_id']):{'title':p['title'],'abstract':p['abstract']} for p in master if str(p['paper_id']) in needed}
    from .nl_labeled import NAMES
    payload=json.dumps({'queries':queries,'rankings':ranks,'qrels':qrels,'papers':papers,'names':NAMES},ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>자연어 추천 평가 라벨 검토</title>
<style>body{font:15px system-ui,sans-serif;color:#172b42;margin:30px;background:#f4f7fa}header,section{background:white;padding:22px;border-radius:12px;margin:14px 0}select{padding:10px;font-size:15px;width:100%}h1{font-size:25px}p{line-height:1.6}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:15px}article{border-bottom:1px solid #dde4eb;padding:14px 0}article:last-child{border:0}.badge{border-radius:5px;padding:3px 8px;color:white;background:#728396}.g2{background:#008878}.g1{background:#b87916}.reason{font-size:13px;color:#53647a}summary{cursor:pointer}details p{font-size:13px}h2{font-size:18px}.note{color:#53647a;font-size:13px}#pool{max-height:650px;overflow:auto}</style>
<header><h1>2차 실험 · 자연어 추천 평가 라벨 검토</h1><p>GPT6 Luna의 제목·초록 기반 관련성 판정: 0 무관 / 1 부분·배경 관련 / 2 직접 관련</p><p class="note">합성 질의·LLM 판정 라벨이며 사람 정답이 아닙니다. 모든 방식의 Top-20 합집합과 무작위 대조군을 판정했습니다. 소재 논문은 추천에서 제외했습니다.</p><select id="pick"></select><p id="query"></p><p class="note" id="meta"></p><label>표시 개수 <select id="depth" style="width:100px"><option>10</option><option>20</option><option>5</option></select></label></header>
<div id="grid"></div><section><details><summary>전체 판정 풀과 근거 보기</summary><div id="pool"></div></details></section>
<script id="data" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
D.queries.forEach((q,i)=>{const o=el('option',q.query_id+' · '+q.language+' · '+q.query);o.value=i;$('pick').append(o)});
function card(pid,rank,labels){const p=D.papers[pid],l=labels[pid],a=el('article');a.append(el('span','관련성 '+l.relevance,'badge g'+l.relevance),el('p',(rank?rank+'. ':'')+p.title),el('p',l.reason,'reason'));const detail=el('details');detail.append(el('summary','초록 · ID 보기'),el('p',p.abstract),el('p','paper_id: '+pid));a.append(detail);return a;}
function render(){const i=+$('pick').value,q=D.queries[i],r=D.rankings.find(x=>x.query_id===q.query_id),v=D.qrels.find(x=>x.query_id===q.query_id),labels=Object.fromEntries(v.labels.map(x=>[x.paper_id,x]));$('query').textContent=q.query;$('meta').textContent=q.language+' · 카테고리 '+q.category+' · 판정 '+v.labels.length+'건';$('grid').replaceChildren();for(const [arm,name] of Object.entries(D.names)){const s=el('section');s.append(el('h2',name));r.rankings[arm].slice(0,+$('depth').value).forEach((p,j)=>s.append(card(p,j+1,labels)));$('grid').append(s)}$('pool').replaceChildren();v.labels.slice().sort((a,b)=>b.relevance-a.relevance).forEach(x=>$('pool').append(card(x.paper_id,0,labels)));} $('pick').onchange=render;$('depth').onchange=render;render();</script></html>'''
    (OUT/'label_review.html').write_text(template.replace('PAYLOAD',payload),encoding='utf-8')
    print(OUT/'label_review.html')

if __name__=='__main__':
    main()
