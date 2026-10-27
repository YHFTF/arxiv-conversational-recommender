"""Portable, self-contained qualitative report (no CDN or server)."""
import json
from pathlib import Path


def render(path, payload):
    template = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>추천 방식 비교 · Benchmark v3</title>
<style>
:root{font-family:Segoe UI,Malgun Gothic,sans-serif;color:#e7ecf4;background:#101620;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:28px 32px;border-bottom:1px solid #344056;background:#172130}h1{margin:0 0 8px;font-size:25px}p{line-height:1.6}.muted{color:#b2bfd3;font-size:13px}.notice{padding:12px 16px;border-left:3px solid #e3af59;background:#2b261c;color:#f1d9ac;margin:16px 0}main{padding:24px 32px}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:18px 0}select,input{padding:10px;background:#182335;color:inherit;border:1px solid #536078;border-radius:6px}select{max-width:100%;min-width:280px}.grid{display:grid;grid-template-columns:repeat(4,minmax(270px,1fr));gap:16px;align-items:start;overflow-x:auto;padding-bottom:16px}.column{min-width:270px}.method{position:sticky;top:0;background:#172130;padding:14px;border-radius:8px;border-top:3px solid #61b8c4;z-index:1}.method h2{font-size:17px;margin:0 0 8px}.card{background:#192333;border:1px solid #344056;border-radius:8px;padding:15px;margin-top:12px}.card.shared{border-left:3px solid #61b8c4}.title{font-size:15px;font-weight:600;line-height:1.5;margin:8px 0}.rank{font-weight:700;color:#8dd7e0}.up{color:#8bd8ab}.down{color:#f5aaa0}.pill{display:inline-block;padding:3px 6px;border-radius:4px;background:#28374b;font-size:11px;margin:3px 4px 0 0}summary{cursor:pointer;color:#91cced;font-size:13px;margin-top:12px}details p{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}.score{font-size:12px;color:#bdcadc;line-height:1.7}.empty{padding:25px;color:#b2bfd3}footer{padding:24px 32px;color:#a4b4ca;font-size:12px}button{padding:9px 14px;background:#255465;color:white;border:0;border-radius:6px;cursor:pointer}@media(max-width:750px){header,main{padding:18px}.grid{grid-template-columns:repeat(4,280px)}}
</style>
<header><div class="muted">BENCHMARK V3 · 정답 없는 정성 비교</div><h1>같은 질의, 네 가지 추천 방식</h1>
<div id="source" class="muted"></div><div class="notice" id="notice"></div><div id="stats" class="muted"></div></header>
<main><div class="controls"><label>질의 <select id="query"></select></label><label>결과에서 찾기 <input id="filter" placeholder="제목 또는 초록"></label><button id="download">비교 JSON 저장</button></div>
<p class="muted">상승·하락은 임베딩만 사용한 전체 순위 대비 변화입니다. 청록색 테두리는 임베딩 기준선의 Top-K에도 있는 논문입니다. 점수는 관련성 확률이나 평가 점수가 아닙니다.</p>
<div class="grid" id="grid"></div></main><footer>새로운 질의를 입력하려면 프로젝트의 compare-recommendations.bat을 실행하세요. 이 파일은 생성 당시 결과를 담은 보고서입니다.</footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent);
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
document.getElementById('source').textContent=data.input_description+' · '+data.model;
document.getElementById('notice').textContent=data.limitations.join(' / ');
const gs=data.graph_stats;document.getElementById('stats').textContent=`논문 ${gs.papers.toLocaleString()}편 · 인용 연결 논문 ${gs.papers_with_citations.toLocaleString()}편 · 저자 연결 논문 ${gs.papers_with_authors.toLocaleString()}편 · 저자 ${gs.authors.toLocaleString()}명 · 그래프 가중치 ${data.config.graph_weight}`;
const select=document.getElementById('query');data.queries.forEach((q,i)=>{const o=el('option',q.query);o.value=i;select.append(o);});
function draw(){const q=data.queries[Number(select.value)],filter=document.getElementById('filter').value.toLowerCase();const grid=document.getElementById('grid');grid.replaceChildren();
for(const [key,method] of Object.entries(q.methods)){const col=el('section',undefined,'column'),head=el('div',undefined,'method');head.append(el('h2',method.name),el('div',`기준선과 ${method.overlap_with_baseline}/${data.top_k}편 중복`,'muted'));if(!method.converged)head.append(el('div','전파가 수렴하지 않았습니다','down'));col.append(head);
const shown=method.rows.filter(r=>(r.title+' '+r.abstract).toLowerCase().includes(filter));
for(const r of shown){const card=el('article',undefined,'card'+(r.baseline_rank<=data.top_k?' shared':''));const rank=el('div',`#${r.rank}`,'rank');if(key!=='text'){rank.append(document.createTextNode('  '),el('span',r.rank_change>0?`↑ ${r.rank_change}`:r.rank_change<0?`↓ ${-r.rank_change}`:'변동 없음',r.rank_change>=0?'up':'down'));}card.append(rank,el('div',r.title,'title'),el('div',`논문 ID ${r.paper_id} · 기준선 #${r.baseline_rank}`,'muted'));
card.append(el('div',`cosine ${r.cosine.toFixed(4)} · 최종 점수 ${r.score.toFixed(6)}`,'score'));
const badges=el('div');badges.append(el('span',`인용 이웃 ${r.citation_neighbors}`,'pill'),el('span',`연결 저자 ${r.graph_authors}`,'pill'));card.append(badges);
const detail=el('details');detail.append(el('summary','초록 · D/T/M · 점수 상세'),el('p',r.abstract||'초록 없음'));
detail.append(el('p','D/T/M: '+JSON.stringify(r.dtm)),el('p',`텍스트 분포 ${r.text_prior.toFixed(8)} / 그래프 분포 ${r.graph_probability.toFixed(8)}`));card.append(detail);col.append(card);}if(!shown.length)col.append(el('div','표시할 결과 없음','empty'));grid.append(col);}}
select.addEventListener('change',draw);document.getElementById('filter').addEventListener('input',draw);
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};draw();
</script></html>'''
    # Prevent a title/abstract containing </script> from escaping the JSON data block.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    Path(path).write_text(template.replace("__PAYLOAD__", encoded), encoding="utf-8")
