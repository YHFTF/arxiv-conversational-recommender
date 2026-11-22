"""Export experiment 1 figures from saved benchmark results (no model rerun)."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'output/citation_benchmark_run1'
OUT = DATA/'figures'
OUT.mkdir(exist_ok=True)
scores=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
deltas=json.loads((DATA/'paired_deltas.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#CDD4DE','text.color':'#172B42','axes.labelcolor':'#46566B',
 'xtick.color':'#46566B','ytick.color':'#46566B','svg.fonttype':'none','pdf.fonttype':42})
ARMS=['graph_only','ogbn','dtm']
NAMES=['인용 GNN (속성 없음)','OGBN 128차원 + GNN','D/T/M Qwen + GNN']
COLORS=['#B7C1CD','#50759E','#008878']
METRICS=[('recall','Recall'),('ndcg','NDCG'),('mrr','MRR')]

def save(fig,name):
 for suffix in ('png','svg','pdf'):
  fig.savefig(OUT/f'{name}.{suffix}',dpi=240,facecolor='white')
 plt.close(fig)

def bars(category=False):
 fig,axes=plt.subplots(1,3,figsize=(14,6.3))
 fig.subplots_adjust(left=.065,right=.98,bottom=.27,top=.69,wspace=.28)
 fig.text(.055,.94,'1차 실험 | D/T/M 기반 논문 표현의 인용 예측 유용성',fontsize=20,weight='bold')
 subtitle='카테고리별 평균을 동일 가중한 결과 · 평가 소스가 있는 38개 카테고리' if category else '전체 논문 후보에서 숨겨진 인용 검색 · 테스트 소스 6,427편'
 fig.text(.055,.875,subtitle,fontsize=12,color='#53647A')
 fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(COLORS,NAMES)],loc='upper left',bbox_to_anchor=(.05,.83),ncol=3,frameon=False,fontsize=11)
 for ax,(metric,title) in zip(axes,METRICS):
  key=('category_macro_' if category else '')+metric
  for j,(arm,color) in enumerate(zip(ARMS,COLORS)):
   values=[scores[arm]['test'][str(k)][key] for k in (10,20)]
   xx=np.array([0,1])+(j-1)*.25
   ax.bar(xx,values,width=.23,color=color,zorder=3)
   for x,y in zip(xx,values):
    label=f'{y*100:.2f}%' if metric=='recall' else f'{y:.4f}'
    ax.text(x,y+.004,label,ha='center',va='bottom',fontsize=9,weight='bold' if arm=='dtm' else 'normal')
  maxval=max(scores[a]['test'][str(k)][key] for a in ARMS for k in (10,20))
  ax.set_ylim(0,maxval*1.28)
  ax.set_xticks([0,1],['@10','@20'])
  ax.set_title(title,fontsize=14,weight='bold',pad=13)
  ax.set_ylabel('점수 (높을수록 좋음)')
  ax.grid(axis='y',color='#E5EAF0',zorder=0)
  if metric=='recall':
   from matplotlib.ticker import PercentFormatter
   ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
  if not category:
   r=(scores['dtm']['test']['20'][metric]/scores['ogbn']['test']['20'][metric]-1)*100
   ax.text(.5,-.23,f'@20 기준선 대비 +{r:.1f}%',transform=ax.transAxes,ha='center',color=COLORS[-1],weight='bold',fontsize=12)
 fig.text(.055,.09,'동일 인용 분할·GNN 구조·평가 후보 사용 | 논문 쌍 단위 8:1:1 분할 | seed=42, 최대 200 epoch',fontsize=10,color='#53647A')
 fig.text(.055,.045,'인용 예측에 대한 유용성 평가이며, 자연어 추천 품질을 직접 측정한 결과는 아님.',fontsize=10,color='#53647A')
 save(fig,'experiment1_category_macro' if category else 'experiment1_performance')

bars()
bars(True)
fig,ax=plt.subplots(figsize=(11,6.4))
fig.subplots_adjust(left=.19,right=.94,bottom=.24,top=.77)
fig.text(.075,.93,'1차 실험 | OGBN 대비 D/T/M의 성능 개선량',fontsize=20,weight='bold')
fig.text(.075,.865,'점: 평균 차이 · 선: 논문 단위 paired bootstrap 95% 신뢰구간 (2,000회)',fontsize=11,color='#53647A')
keys=['recall@10','recall@20','ndcg@10','ndcg@20','mrr@10','mrr@20']
for i,key in enumerate(keys):
 d=deltas[key];v=d['dtm_minus_ogbn'];lo,hi=d['paired_bootstrap_95']
 ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[-1],capsize=5,markersize=7,linewidth=2)
 ax.text(hi+.001,i,f'+{v:.4f}  [{lo:.4f}, {hi:.4f}]',va='center',fontsize=10)
ax.axvline(0,color='#7C8898',linestyle='--',linewidth=1)
ax.set_yticks(range(6),['Recall@10','Recall@20','NDCG@10','NDCG@20','MRR@10','MRR@20'])
ax.invert_yaxis();ax.set_xlim(-.002,.105)
ax.set_xlabel('D/T/M - OGBN（絶対スコア差）'.replace('絶対スコア差','절대 점수 차이'),labelpad=12)
ax.grid(axis='x',color='#E5EAF0')
fig.text(.075,.11,'모든 구간이 0보다 높음: 이번 실행에서 D/T/M 기반 모델의 일관된 개선을 관찰.',fontsize=11,color=COLORS[-1],weight='bold')
fig.text(.075,.055,'단일 분할·학습 seed=42 결과. 구간은 평가 소스의 표본 변동성을 나타내며 학습 seed 변동성은 포함하지 않음.',fontsize=9.5,color='#53647A')
save(fig,'experiment1_improvement_ci')
with (OUT/'experiment1_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['arm','k','recall','ndcg','mrr','category_macro_recall','category_macro_ndcg','category_macro_mrr'])
 for arm in ARMS:
  for k in (10,20):
   r=scores[arm]['test'][str(k)];w.writerow([arm,k]+[r[m] for m in ('recall','ndcg','mrr','category_macro_recall','category_macro_ndcg','category_macro_mrr')])
(OUT/'caption_ko.txt').write_text('그림 1. D/T/M 기반 논문 표현의 인용 링크 예측 성능. 동일한 인용 그래프 분할과 GNN을 사용해 OGBN 속성, D/T/M의 Qwen3-Embedding-4B 임베딩, 속성 없는 모델을 비교하였다. 논문 쌍을 8:1:1로 분할하고 16,000편의 코퍼스에서 테스트 소스 6,427편을 평가하였다. D/T/M 모델은 OGBN 기준선 대비 Recall@20에서 38.8%, NDCG@20에서 49.7%의 상대 향상을 보였다. 이는 자연어 추천에 D/T/M 표현을 도입하는 근거를 제공하지만, 자연어 추천 품질의 직접적인 향상을 입증하는 결과는 아니다. 단일 seed=42의 결과이며 입력 모델·차원과 입력 투영 파라미터 수가 다르다.',encoding='utf-8')
print(OUT)
