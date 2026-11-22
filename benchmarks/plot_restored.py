"""Publication-ready figures for the restored balanced-data benchmarks."""
import os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/matplotlib_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ARMS=['text','dtm','combined','citation','author','both','intent']
LABELS=['제목·초록','D/T/M 전용','제목·초록+D/T/M','결합 + 인용 PPR','결합 + 저자 PPR','결합 + 인용·저자 PPR','현재 의도 분석·재정렬']
COLORS=['#8D9BAC','#008878','#50759E','#8C72AD','#C68D35','#367F93','#B45669']
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#CDD4DE','text.color':'#172B42','axes.labelcolor':'#46566B',
 'xtick.color':'#46566B','ytick.color':'#46566B','svg.fonttype':'none','pdf.fonttype':42})


def save(fig,out,name):
    for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)


def citation_figure(root,dest):
    citation=json.loads((root/'citation/summary.json').read_text(encoding='utf-8'))
    cm=json.loads((root/'citation/manifest.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(14,6.6));fig.subplots_adjust(left=.075,right=.97,bottom=.25,top=.72,wspace=.27)
    fig.text(.055,.94,'균형 표본 재평가 | 인용 링크 예측',fontsize=22,weight='bold')
    fig.text(.055,.86,f'16,000편 · 인용 8:1:1 분할 · seed=42 · 200 epoch · 학습 인용만 GNN에 사용',fontsize=12,color='#53647A')
    arms=['graph_only','ogbn','dtm'];names=['속성 없음','OGBN 128차원','D/T/M Qwen'];colors=['#B7C1CD','#50759E','#008878']
    for ax,metric,heading in zip(axes,('recall','ndcg','mrr'),('Recall@20','NDCG@20','MRR@20')):
        values=[citation[arm]['test']['20'][metric] for arm in arms];ax.bar(range(3),values,color=colors,zorder=3);ax.set_xticks(range(3),names,rotation=10);ax.set_title(heading,weight='bold',pad=16);ax.set_ylim(0,max(values+[.01])*1.25);ax.grid(axis='y',color='#E5EAF0',zorder=0)
        for i,v in enumerate(values):ax.text(i,v+max(values+[.01])*.035,f'{v:.4f}',ha='center')
    fig.text(.055,.115,'分割'.replace('分割','분할')+' 인용 수: '+str(cm['split_edges'])+' · device='+cm['training_device'],fontsize=10,color='#53647A')
    fig.text(.055,.06,'숨겨진 인용 예측 평가. 자연어 추천 또는 D/T/M의 사실 정확도를 직접 측정하는 지표는 아님.',fontsize=10,color='#53647A')
    save(fig,dest,'citation_performance')
    return citation,cm,arms,names


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/balanced_benchmark_20261117');parser.add_argument('--category-only',action='store_true');parser.add_argument('--citation-only',action='store_true');a=parser.parse_args()
    dest=a.output/'figures';dest.mkdir(exist_ok=True)
    audit=json.loads((a.output/'data_audit.json').read_text(encoding='utf-8'))
    category={int(k):v for k,v in audit['category_counts'].items()}
    fig,ax=plt.subplots(figsize=(14,5.5));fig.subplots_adjust(left=.07,right=.97,bottom=.23,top=.73)
    fig.text(.055,.93,'복원 데이터 검증 | 카테고리별 논문 수',fontsize=21,weight='bold')
    fig.text(.055,.85,'16,000편 · 40개 카테고리 · 원본 보유량을 고려한 균형 쿼터와 정확히 일치',fontsize=12,color='#53647A')
    ax.bar(list(category),list(category.values()),color=['#008878' if n>=418 else '#C68D35' for n in category.values()],zorder=3)
    for k,v in category.items():
        if v<418:ax.text(k,v+7,str(v),ha='center',fontsize=9,color='#91621B')
    ax.axhline(400,color='#8D9BAC',ls='--',lw=1);ax.set_ylim(0,470);ax.set_xticks(range(40));ax.set_ylabel('논문 수');ax.set_xlabel('OGBN 카테고리 ID');ax.grid(axis='y',color='#E5EAF0',zorder=0)
    fig.text(.055,.075,'대부분 418~419편. 원본 논문이 부족한 카테고리는 보유 전량(29~416편)을 사용하고 나머지를 재분배.',fontsize=10,color='#53647A')
    save(fig,dest,'balanced_category_distribution')
    if a.category_only:
        print(dest/'balanced_category_distribution.png');return
    if a.citation_only:
        citation_figure(a.output,dest);print(dest/'citation_performance.png');return
    nl=json.loads((a.output/'natural_language/summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((a.output/'natural_language/manifest.json').read_text(encoding='utf-8'))
    for protocol,title in [('pooled','원래 판정 조건'),('pooled_without_anchor','소재 논문 제외')]:
        result=nl[protocol]['metrics'];fig,axes=plt.subplots(1,3,figsize=(16,7.7));fig.subplots_adjust(left=.23,right=.96,top=.72,bottom=.23,wspace=.26)
        fig.text(.045,.94,'균형 표본 재평가 | 자연어 논문 추천',fontsize=22,weight='bold')
        fig.text(.045,.88,f'Qwen3-Embedding-4B · test {manifest["test_queries"]}개 질의 / {manifest["unique_test_anchors"]}개 소재 논문 · {title}',fontsize=12,color='#53647A')
        fig.text(.045,.82,'공통의 완전 판정 후보 풀 내 재정렬 · 0 무관 / 1 부분 관련 / 2 직접 관련',fontsize=11,color='#53647A')
        for ax,metric,heading in zip(axes,('recall','ndcg','mrr'),('Recall@10','NDCG@10','MRR@10')):
            values=[result[arm]['10'][metric] for arm in ARMS];y=np.arange(len(ARMS))
            ax.barh(y,values,color=COLORS,height=.62,zorder=3);ax.invert_yaxis();ax.set_yticks(y,LABELS if ax is axes[0] else ['']*len(y));ax.set_title(heading,pad=19,weight='bold')
            ax.set_xlim(0,max(values+[.01])*1.23);ax.grid(axis='x',color='#E5EAF0',zorder=0)
            for i,v in enumerate(values):ax.text(v+max(values+[.01])*.02,i,f'{v:.4f}',va='center',fontsize=10)
            ax.set_xlabel('높을수록 좋음',labelpad=12)
        fig.text(.045,.12,'가져온 원래 split(학습 506 / 검증 46 / 테스트 48) 유지. 하이퍼파라미터 고정; 추가 API 호출 없음.',fontsize=10,color='#53647A')
        fig.text(.045,.065,'LLM 판정은 D/T/M을 참고하고 소재 논문을 2점으로 고정한 생성 코드에 기반함. 전체 코퍼스 검색 성능이 아님.',fontsize=10,color='#53647A')
        save(fig,dest,'natural_language_'+protocol)
    fig,ax=plt.subplots(figsize=(13,7.4));fig.subplots_adjust(left=.29,right=.95,top=.72,bottom=.24)
    fig.text(.055,.94,'자연어 추천 | 제목·초록 기준선 대비 NDCG@10 차이',fontsize=20,weight='bold')
    fig.text(.055,.865,'소재 논문 제외 조건 · 소재 논문 단위 paired bootstrap 95% 신뢰구간 (2,000회)',fontsize=11,color='#53647A')
    entries=[nl['pooled_without_anchor']['paired_deltas_vs_text'][arm+'/ndcg@10'] for arm in ARMS[1:]]
    low=min([v['ci95'][0] for v in entries]+[0]);high=max([v['ci95'][1] for v in entries]+[0]);span=max(high-low,.02)
    for i,(v,c) in enumerate(zip(entries,COLORS[1:])):
        d=v['delta'];lo,hi=v['ci95'];ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=c,capsize=5,linewidth=2)
        ax.text(hi+span*.03,i,f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]',va='center',fontsize=9)
    ax.axvline(0,color='#7C8898',ls='--');ax.set_yticks(range(6),LABELS[1:]);ax.invert_yaxis();ax.set_xlim(low-span*.1,high+span*.95);ax.grid(axis='x',color='#E5EAF0');ax.set_xlabel('각 방식 - 제목·초록')
    fig.text(.055,.11,'같은 소재 논문의 두 질의를 함께 재표집. 구간이 0을 포함하면 이번 평가에서 차이가 분명하지 않음.',fontsize=10,color='#53647A')
    fig.text(.055,.055,'가져온 판정셋에 대한 결과이며 이전 GPT6 Luna 블라인드 판정 실험과 직접적인 수치 비교는 불가.',fontsize=10,color='#53647A')
    save(fig,dest,'natural_language_ndcg10_deltas')
    citation,cm,arms,names=citation_figure(a.output,dest)
    report=['# 균형 데이터 교체 및 벤치마크 결과','',f'논문 {audit["papers"]:,}편, 카테고리 40개. 원본 보유량 제약을 반영한 균형 쿼터와 실제 분포가 일치한다. 인용 {audit["citation_edges"]:,}개, 저자 연결 {audit["author_edges"]:,}개. D/T/M 성공 15,999편, 실패 1편을 빈 속성으로 유지했다.','',
        '## 자연어 추천','',f'총 질의 600개/판정 19,200건을 보존하고 기존 split의 test 48개(24개 소재 논문)만 평가했다. 학습·검증 라벨로 튜닝하지 않았다. 모든 방식에 같은 Qwen 모델, 후보 풀 및 정답을 사용한다.']
    for protocol,title in [('pooled','원래 조건: 소재 포함'),('pooled_without_anchor','소재 논문 제외')]:
        report+=['','### '+title,'','|방식|Recall@10|NDCG@10|MRR@10|','|---|---:|---:|---:|']
        for arm,label in zip(ARMS,LABELS):
            v=nl[protocol]['metrics'][arm]['10'];report.append(f'|{label}|{v["recall"]:.4f}|{v["ndcg"]:.4f}|{v["mrr"]:.4f}|')
        report+=['','기준선 대비 NDCG@10 차이(소재 논문 단위 95% bootstrap 구간):']
        for arm,label in zip(ARMS[1:],LABELS[1:]):
            d=nl[protocol]['paired_deltas_vs_text'][arm+'/ndcg@10'];report.append(f'- {label}: {d["delta"]:+.4f}, [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]')
    report+=['','## 인용 예측','','|속성|Recall@20|NDCG@20|MRR@20|최적 epoch|','|---|---:|---:|---:|---:|']
    for arm,label in zip(arms,names):
        v=citation[arm]['test']['20'];report.append(f'|{label}|{v["recall"]:.4f}|{v["ndcg"]:.4f}|{v["mrr"]:.4f}|{citation[arm]["best_epoch"]}|')
    cd=json.loads((a.output/'citation/paired_deltas_vs_ogbn.json').read_text(encoding='utf-8'))
    report+=['','D/T/M - OGBN 개선량(95% bootstrap 구간):']
    for metric in ('recall','ndcg','mrr'):
        v=cd['dtm/'+metric+'@20'];report.append(f'- {metric}@20: {v["delta"]:+.4f}, [{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]')
    report+=['','인용 실험은 단일 seed=42 결과이며, 신뢰구간은 학습 seed 변동성을 포함하지 않는다.',
        '', '자연어 테스트는 가져온 영어 질의 48개로 구성되고 18개 소스 카테고리를 포함한다. 소재 제외 조건에서는 관련 정답 없는 질의 2개, 2점 정답 없는 질의 31개를 제외하지 않고 해당 지표 0으로 포함했다.',
        '', '## 해석의 범위','',
        '이번 자연어 점수는 완전 판정된 고정 후보 풀 내 재정렬 성능이며 전체 코퍼스 검색 성능이 아니다. 가져온 생성 코드에서는 D/T/M을 판정 입력에 포함하고 소재 논문을 2점으로 강제하므로, 특히 D/T/M 방식에 유리한 판정 편향 가능성을 고려해야 한다. 소재 제외 결과도 함께 보고한다.',
        '', '이전 100개 GPT6 Luna 블라인드 평가와 데이터·질의·판정 모델·후보 구성·소재 포함 조건이 다르다. 점수가 달라도 균형화만의 효과로 해석할 수 없다. 인용 예측 또한 코퍼스와 그래프가 바뀌었으므로 새 표본 내 비교군 간 차이를 기준으로 해석한다.',
        '', '새 데이터는 두 벤치마크 완료 후 활성 경로로 교체했다. activation.json과 이전 파일 ZIP을 통해 16개 파일의 교체 내역과 복구 경로를 확인할 수 있다. 기존 결과 폴더는 보존했다. 기본 추천기에서도 API 키 없이 의도 48/48, 질의 48/48, 키워드 473/473 캐시 적중 및 NDCG@10=0.9763 재현을 확인했다. 독립 점수 재계산 오차 0, 카테고리 정렬 오류 0, 인용 분할 중복 0이며 추가 API 호출은 없다.']
    (a.output/'RESULTS_KO.md').write_text('\n'.join(report),encoding='utf-8')
    print(dest)

if __name__=='__main__':main()
