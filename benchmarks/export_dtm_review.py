"""Export a diverse train-only review queue; no prefilled relevance labels."""
from pathlib import Path
import json,random
from .core import read,write


def main():
    out=Path(__file__).resolve().parents[1]/'output/balanced_benchmark_20261117/guarded_dtm_graph'
    rows=read(out/'unjudged_train_review.json');by={}
    for r in rows:by.setdefault(r['query_id'],[]).append(r)
    keys=sorted(by);random.Random(42).shuffle(keys);sample=[by[q][0] for q in keys[:120]]
    write(out/'review_sample.json',sample)
    payload=json.dumps(sample,ensure_ascii=False).replace('<',chr(92)+'u003c')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>D/T/M 검색 train 판정</title>
<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;background:#f6f7f9}article{background:white;padding:24px;border-radius:12px;margin:16px 0}button{padding:10px 16px;margin:5px;cursor:pointer}pre{white-space:pre-wrap;font:inherit;line-height:1.65}.muted{color:#555}#note{width:95%;height:65px}#status{font-weight:bold}</style>
<h1>실제 검색 후보 판정</h1><p>train 질의만 포함합니다. 기존 정답이 없는 후보이며 기본값은 미판정입니다. 0=무관, 1=부분 관련, 2=직접 관련. 초록만으로 조건을 확인할 수 없다면 미판정으로 남기고 메모하세요. 저장은 이 브라우저에만 유지되며 외부 전송은 없습니다.</p>
<p id="status"></p><button onclick="move(-1)">이전</button><button onclick="move(1)">다음</button><button onclick="download()">판정한 항목 JSON 내보내기</button>
<article><h2 id="query"></h2><p id="id" class="muted"></p><h3 id="title"></h3><pre id="abstract"></pre><div id="buttons"></div><textarea id="note" placeholder="판정 근거 또는 확인 불가 조건" oninput="saveNote()"></textarea></article>
<script>const rows=PAYLOAD;let pos=0;const key='dtm-train-review-v1';let labels=JSON.parse(localStorage.getItem(key)||'{}');
function ident(r){return r.query_id+':'+r.paper_id}function persist(){localStorage.setItem(key,JSON.stringify(labels))}
function show(){const r=rows[pos],v=labels[ident(r)]||{grade:null,note:''};document.getElementById('query').textContent=r.query;document.getElementById('id').textContent=r.query_id+' / '+r.paper_id;document.getElementById('title').textContent=r.title;document.getElementById('abstract').textContent=r.abstract;document.getElementById('note').value=v.note;document.getElementById('status').textContent=(pos+1)+' / '+rows.length+' · 판정 '+Object.values(labels).filter(v=>v.grade!==null).length+'개';const box=document.getElementById('buttons');box.replaceChildren();for(const [grade,name] of [[null,'미판정'],[0,'0 무관'],[1,'1 부분 관련'],[2,'2 직접 관련']]){const b=document.createElement('button');b.textContent=(v.grade===grade?'✓ ':'')+name;b.onclick=()=>{labels[ident(r)]={grade,note:document.getElementById('note').value};persist();show()};box.append(b)}}
function saveNote(){const r=rows[pos],v=labels[ident(r)]||{grade:null};v.note=document.getElementById('note').value;labels[ident(r)]=v;persist()}
function move(d){pos=Math.max(0,Math.min(rows.length-1,pos+d));show()}
function download(){const selected=rows.filter(r=>labels[ident(r)]&&labels[ident(r)].grade!==null).map(r=>({query_id:r.query_id,paper_id:r.paper_id,grade:labels[ident(r)].grade,reason:labels[ident(r)].note,split:'train',source:'manual-review'}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'}));a.download='dtm-reviewed-train.json';a.click();URL.revokeObjectURL(a.href)}show();</script></html>'''.replace('PAYLOAD',payload)
    (out/'review.html').write_text(html,encoding='utf-8')
    print('Saved',out/'review.html','pairs',len(sample))


if __name__=='__main__':main()
