"""Render the temporal zero-edge cold-source results and independent checks."""
import os,json,argparse,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/matplotlib_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .core import read,write,file_hash
from .cold_start import ARMS,ranked_metrics

NAMES={'text_cosine':'제목·초록 코사인','dtm_cosine':'DTM 코사인','combined_cosine':'결합 코사인',
 'ogbn_gnn':'OGBN 속성 + GNN','text_gnn':'제목·초록 + GNN','dtm_gnn':'DTM + GNN','combined_gnn':'결합 + GNN'}
COLORS=['#AEB8C4','#8AC7BA','#97B3D1','#B6A5C9','#50759E','#008878','#7C72A6']
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#CDD4DE','text.color':'#172B42','axes.labelcolor':'#46566B',
 'xtick.color':'#46566B','ytick.color':'#46566B','svg.fonttype':'none','pdf.fonttype':42})


def save(fig,folder,name):
    for ext in ('png','pdf','svg'):fig.savefig(folder/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)


def verify(out):
    s=read(out/'split.json');parts=s['partition_global_rows'];years=s['years'];warm=parts['train'];v=set(parts['validation']);t=set(parts['test'])
    assert not set(warm)&v and not set(warm)&t and not v&t
    assert all(years[g]<=2017 for g in warm) and all(years[g]==2018 for g in v) and all(years[g]>=2019 for g in t)
    assert all(0<=a<len(warm) and 0<=b<len(warm) for a,b in s['train_edges_local'])
    old_nodes={s['node_indices'][g] for g in warm};test_nodes={s['node_indices'][g] for g in t}
    result=read(out/'summary.json');max_error=0.
    for seed in result['seeds']:
        for arm in ARMS:
            rows=read(out/f'seed_{seed}'/arm/'per_paper.json')
            for r in rows:
                assert r['source_node_idx'] in test_nodes
                assert set(r['ranking_node_indices'])<=old_nodes and set(r['truth_node_indices'])<=old_nodes
                truth=set(r['truth_node_indices']);rank=r['ranking_node_indices'];k=r['k']
                hits=[i for i,n in enumerate(rank[:k]) if n in truth]
                ideal=sum(1/math.log2(i+2) for i in range(min(k,len(truth))))
                expected=sum(1/math.log2(i+2) for i in hits)/ideal
                max_error=max(max_error,abs(expected-r['ndcg']))
            for k in (10,20):
                for metric in ('recall','ndcg','mrr'):
                    measured=np.mean([r[metric] for r in rows if r['k']==k])
                    stored=next(v for v in result['arms'][arm]['by_seed'] if v['seed']==seed)['test'][str(k)][metric]
                    assert abs(measured-stored)<1e-12
    assert max_error<1e-12
    info={'node_partitions_disjoint':True,'publication_year_cutoffs_verified':True,
      'cold_nodes_absent_from_train_graph':True,'negative_candidate_space':'train-local node indices only',
      'ranking_and_truth_targets_are_old_papers_only':True,'metric_recalculation_max_error':max_error,
      'seeds':result['seeds'],'additional_api_calls':0}
    write(out/'verification.json',info)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/cold_start_temporal_20261117');args=parser.parse_args();out=args.output
    verify(out);summary=read(out/'summary.json');audit=read(out/'data_audit.json');figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,7.6));fig.subplots_adjust(left=.24,right=.96,top=.73,bottom=.23,wspace=.26)
    fig.text(.045,.94,'콜드스타트 | 인용 연결 0개인 신규 논문의 참고문헌 검색',fontsize=21,weight='bold')
    fig.text(.045,.87,f'연도 분리: 학습 <=2017 / 검증 2018 / 테스트 >=2019 · 평가 소스 {audit["eligible_sources"]["test"]:,}편 · 후보 {audit["candidate_count"]:,}편',fontsize=11,color='#53647A')
    fig.text(.045,.81,'막대: 3개 학습 seed의 평균 · 오차막대: seed 간 표준편차 · 정답은 숨긴 실제 인용',fontsize=11,color='#53647A')
    for ax,metric,title in zip(axes,('recall','ndcg','mrr'),('Recall@20','NDCG@20 [주 지표]','MRR@20')):
        means=[summary['arms'][arm]['test']['20'][metric]['mean'] for arm in ARMS]
        std=[summary['arms'][arm]['test']['20'][metric]['seed_std'] for arm in ARMS]
        ax.barh(range(len(ARMS)),means,xerr=std,color=COLORS,height=.62,capsize=3,zorder=3)
        ax.set_yticks(range(len(ARMS)),[NAMES[a] for a in ARMS] if ax is axes[0] else ['']*len(ARMS));ax.invert_yaxis()
        ax.set_xlim(0,max(m+s for m,s in zip(means,std))*1.35);ax.set_title(title,pad=16,weight='bold');ax.grid(axis='x',color='#E5EAF0',zorder=0)
        for i,(m,s) in enumerate(zip(means,std)):ax.text(m+s+max(means)*.025,i,f'{m:.4f}',va='center',fontsize=10)
        ax.set_xlabel('높을수록 좋음',labelpad=12)
    fig.text(.045,.115,'신규 논문은 GNN 학습 그래프·양성·음성 샘플에서 완전히 제외. 테스트 논문의 인용은 추론 입력에 주지 않음.',fontsize=10,color='#53647A')
    fig.text(.045,.06,'과거 데이터에서 미관측 노드를 모사한 평가. 신규 논문 자체의 사용자 노출이나 실제 2026년 신간 추천 성능을 뜻하지 않음.',fontsize=10,color='#53647A')
    save(fig,figures,'cold_start_performance')
    fig,ax=plt.subplots(figsize=(10,5.6));fig.subplots_adjust(left=.1,right=.95,bottom=.2,top=.72)
    fig.text(.075,.94,'동일 Qwen·동일 GNN | 학습 seed별 콜드스타트 성능',fontsize=19,weight='bold')
    for arm,color in [('text_gnn','#50759E'),('dtm_gnn','#008878'),('combined_gnn','#7C72A6')]:
        values=[r['test']['20']['ndcg'] for r in summary['arms'][arm]['by_seed']]
        ax.plot(range(len(values)),values,'o-',color=color,label=NAMES[arm],lw=2,ms=7)
    ax.set_xticks(range(len(summary['seeds'])),[str(s) for s in summary['seeds']]);ax.set_xlabel('학습 seed');ax.set_ylabel('NDCG@20');ax.set_ylim(bottom=0);ax.grid(axis='y',color='#E5EAF0');ax.legend(frameon=False)
    fig.text(.075,.065,'노드 분할과 후보는 고정하고 초기화·학습 음성 샘플의 seed만 변경함.',fontsize=10,color='#53647A')
    save(fig,figures,'cold_start_by_seed')
    report=['# 연결 0개 신규 논문: 시간 분리 콜드스타트 평가','',
        f'학습 {audit["node_split"]["train"]:,}편(2017년까지), 검증 {audit["node_split"]["validation"]:,}편(2018년), 테스트 {audit["node_split"]["test"]:,}편(2019년 이후).',
        f'기존 후보를 인용한 테스트 소스 {audit["eligible_sources"]["test"]:,}편, 정답 인용 {audit["held_out_references"]["test"]:,}개. 후보는 학습 논문 {audit["candidate_count"]:,}편 전체이다.',
        '', '새 논문은 학습 그래프와 학습 양성·음성에서 모두 제외했다. 공유 속성 변환과 인용 점수 함수를 기존 논문에서 학습하고, 신규 논문에는 자기 연결만 있는 상태로 적용한다. 테스트 노드의 이웃 전파는 없으며 기존 후보 벡터는 학습 그래프에서 구한다. 추가 API 호출과 임베딩 재생성은 없다.',
        '', '|방식|Recall@20 평균 ± seed 표준편차|NDCG@20 평균 ± seed 표준편차|MRR@20 평균|','|---|---:|---:|---:|']
    for arm in ARMS:
        v=summary['arms'][arm]['test']['20'];report.append(f'|{NAMES[arm]}|{v["recall"]["mean"]:.4f} ± {v["recall"]["seed_std"]:.4f}|{v["ndcg"]["mean"]:.4f} ± {v["ndcg"]["seed_std"]:.4f}|{v["mrr"]["mean"]:.4f}|')
    report+=['','## 동일 Qwen 제목·초록 GNN 대비 차이','', '3개 seed 및 소스 논문을 짝지어 재표집한 95% bootstrap 구간이다.']
    for arm in ARMS:
        if arm=='text_gnn':continue
        d=summary['paired_deltas_vs_text_gnn'][arm+'/ndcg@20'];report.append(f'- {NAMES[arm]}: {d["delta"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]')
    d=summary['paired_deltas_vs_text_gnn']['dtm_gnn/ndcg@20']
    verdict='DTM GNN의 제목·초록 GNN 대비 개선 구간이 양수이다.' if d['ci95'][0]>0 else ('DTM GNN이 제목·초록 GNN보다 낮은 구간이다.' if d['ci95'][1]<0 else 'DTM GNN과 제목·초록 GNN의 차이가 명확하지 않다.')
    report+=['','## 해석','',verdict,
        '', 'DTM 추가 가치는 동일 Qwen·동일 차원의 제목·초록 GNN 및 학습 없는 제목·초록 코사인 기준선과 함께 판단해야 한다. OGBN 128차원 비교만으로 DTM 자체의 기여를 단정하지 않는다.',
        '', '이 실험은 새 논문을 입력하여 기존 관련 논문/참고문헌을 찾는 과제이다. 새 논문을 사용자에게 추천 후보로 노출하는 과제는 평가하지 않는다. 과거 논문을 GNN에 보류한 모사 실험이며 Qwen·DTM 추출 모델이 해당 논문을 사전학습에서 본 적이 없다는 의미는 아니다.',
        '', f'테스트 {audit["unscored_no_old_references"]["test"]}편은 후보 코퍼스 내 알려진 참고문헌이 없어 분모를 만들 수 없으므로 점수를 계산하지 않았다. 노드 분리와 연도는 검증했지만 과거 시점의 실제 인용 관측 시각은 데이터에 없어 완전한 역사적 스냅샷 복원은 아니다. 단일 시간 구간과 3개 seed 결과로 일반적인 신규 논문 추천 우위를 주장할 수 없다.']
    (out/'RESULTS_KO.md').write_text('\n'.join(report),encoding='utf-8')
    write(out/'implementation_sha256.json',{n:file_hash(ROOT/'benchmarks'/n) for n in ('cold_start.py','plot_cold_start.py','citation.py')})
    print(figures)

if __name__=='__main__':main()
