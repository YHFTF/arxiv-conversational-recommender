"""Plot experiment 2 from actual GPT6 Luna judged results."""
import csv
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/matplotlib_config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

DATA=ROOT/'output/nl_benchmark_gpt6_luna_run1'
OUT=DATA/'figures'
COLORS=['#8D9BAC','#008878','#50759E','#8C72AD','#C68D35','#367F93','#B45669']
ARMS=['text','dtm','combined','citation','author','both','intent']
LABELS=['제목·초록','D/T/M 전용','기존 제목·초록+D/T/M','결합 + 인용 PPR','결합 + 저자 PPR','결합 + 인용·저자 PPR','현재 의도 분석·재정렬']
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#CDD4DE','text.color':'#172B42','axes.labelcolor':'#46566B',
 'xtick.color':'#46566B','ytick.color':'#46566B','svg.fonttype':'none','pdf.fonttype':42})


def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    consistency=json.loads((DATA/'judge_consistency.json').read_text(encoding='utf-8'))
    m=summary['metrics']
    y=np.arange(len(ARMS))
    for k in (10,20):
        fig,axes=plt.subplots(1,3,figsize=(16,7.6))
        fig.subplots_adjust(left=.23,right=.96,top=.73,bottom=.22,wspace=.25)
        fig.text(.045,.94,'2차 실험 | 자연어 논문 추천의 관련성 평가',fontsize=22,weight='bold')
        fig.text(.045,.88,f'질의 {summary["queries"]}개 (한국어 50·영어 50) · GPT6 Luna의 관련성 0/1/2 판정 {summary["labels"]:,}건',fontsize=12,color='#53647A')
        fig.text(.045,.825,'0 무관 / 1 부분·배경 관련 / 2 직접 관련 · 동일 질의·정답 풀 · 원본 소재 논문 제외',fontsize=11,color='#53647A')
        for ax,metric,title in zip(axes,('recall','ndcg','mrr'),(f'Known-positive Recall@{k}',f'NDCG@{k}'+(' [주 지표]' if k==10 else ''),f'MRR@{k}')):
            values=[m[a][str(k)][metric] for a in ARMS]
            ax.barh(y,values,color=COLORS,height=.62,zorder=3)
            ax.set_yticks(y,LABELS if ax is axes[0] else ['']*len(y));ax.invert_yaxis()
            ax.set_xlim(0,max(values)*1.23 if max(values)>0 else 1)
            ax.set_title(title,fontsize=13,pad=20,weight='bold')
            for i,v in enumerate(values):
                ax.text(v+max(values)*.025,i,f'{v*100:.1f}%' if metric=='recall' else f'{v:.4f}',va='center',fontsize=10)
            ax.grid(axis='x',color='#E5EAF0',zorder=0)
            if metric=='recall':
                ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.set_xlabel('높을수록 좋음',labelpad=13)
        fig.text(.045,.115,'전체 16,000편에서 검색한 Top-20의 합집합 + 무작위 후보에 판정. Recall의 분모는 판정 풀의 알려진 관련 논문.',fontsize=10,color='#53647A')
        fig.text(.045,.065,'LLM 생성 질의·LLM 판정 라벨이며 사람 정답이 아님. PPR은 학습 없는 전파이며 1차 실험의 GNN과 다름.',fontsize=10,color='#53647A')
        save(fig,f'experiment2_performance_at{k}')
    fig,ax=plt.subplots(figsize=(13,7))
    fig.subplots_adjust(left=.29,right=.93,top=.73,bottom=.25)
    fig.text(.055,.94,'2차 실험 | 제목·초록 기준선 대비 NDCG@10 차이',fontsize=21,weight='bold')
    fig.text(.055,.865,'점: 질의별 평균 차이 · 선: 질의 단위 paired bootstrap 95% 신뢰구간 (2,000회)',fontsize=11,color='#53647A')
    entries=[summary['paired_deltas_vs_text'][a+'/ndcg@10'] for a in ARMS[1:]]
    lows=[v['ci95'][0] for v in entries];highs=[v['ci95'][1] for v in entries]
    span=max(highs+[0])-min(lows+[0]);span=max(span,.01)
    for i,(v,c) in enumerate(zip(entries,COLORS[1:])):
        lo,hi=v['ci95'];d=v['delta']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=c,capsize=5,linewidth=2,markersize=7)
        ax.text(hi+span*.035,i,f'{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]',va='center',fontsize=10)
    ax.axvline(0,color='#7C8898',ls='--',linewidth=1)
    ax.set_yticks(range(6),LABELS[1:]);ax.invert_yaxis()
    ax.set_xlim(min(lows+[0])-span*.1,max(highs+[0])+span*.9)
    ax.grid(axis='x',color='#E5EAF0');ax.set_xlabel('各方式 - 題目·抄録'.replace('各方式 - 題目·抄録','각 방식 - 제목·초록'),labelpad=12)
    fig.text(.055,.12,'0의 오른쪽은 개선, 왼쪽은 하락. 구간이 0을 포함하면 해당 비교의 차이가 분명하지 않음.',fontsize=11,color='#53647A')
    fig.text(.055,.06,'동일 GPT6 Luna의 합성 질의·판정 라벨에 대한 결과. 다중 비교 보정 없음; 사람 평가 및 다른 질의 집합에 대한 일반화는 미검증.',fontsize=9.5,color='#53647A')
    save(fig,'experiment2_ndcg10_improvement_ci')
    fig,ax=plt.subplots(figsize=(13,8))
    fig.subplots_adjust(left=.28,right=.94,top=.76,bottom=.19)
    fig.text(.055,.94,'2차 실험 | 질의 언어별 NDCG@10',fontsize=21,weight='bold')
    fig.text(.055,.87,'한국어 50개·영어 50개: 서로 다른 소재 논문에서 생성한 질의이며 번역 쌍 비교는 아님.',fontsize=11,color='#53647A')
    for j,(lang,color,label) in enumerate((('Korean','#008878','한국어'),('English','#50759E','영어'))):
        values=[m[a]['10']['by_language'][lang]['ndcg'] for a in ARMS]
        ax.barh(y+(j-.5)*.34,values,height=.31,color=color,label=label,zorder=3)
        for i,v in enumerate(values):
            ax.text(v+.006,i+(j-.5)*.34,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.set_xlim(0,1.12)
    ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',color='#E5EAF0')
    ax.set_xlabel('NDCG@10 (높을수록 좋음)')
    fig.text(.055,.07,'판정 모델: GPT6 Luna · 모든 방식에 공통의 제목·초록 기반 0/1/2 라벨 사용',fontsize=10,color='#53647A')
    save(fig,'experiment2_language_breakdown')
    with (OUT/'experiment2_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','name','k','recall','strong_recall','ndcg','mrr'])
        for arm,label in zip(ARMS,LABELS):
            for k in (5,10,20):
                w.writerow([arm,label,k]+[m[arm][str(k)][x] for x in ('recall','strong_recall','ndcg','mrr')])
    text=['# 2차 실험: 자연어 추천 관련성 평가','',f'질의 {summary["queries"]}개, 라벨 {summary["labels"]:,}건. 주 지표 NDCG@10.','',
      '|방식|Recall@10 (known-positive)|NDCG@10|MRR@10|','|---|---:|---:|---:|']
    for a,label in zip(ARMS,LABELS):
        v=m[a]['10'];text.append(f'|{label}|{v["recall"]:.4f}|{v["ndcg"]:.4f}|{v["mrr"]:.4f}|')
    audit=json.loads((DATA/'input_truncation_audit.json').read_text(encoding='utf-8')) if (DATA/'input_truncation_audit.json').exists() else {}
    text+=['',f'입력 길이 감사: {audit}', '',f'라벨 분포: {summary["label_distribution"]}. 관련 정답 없는 질의: {summary["zero_positive_queries"]}.',
      f'10개 질의 재판정 일치율: {consistency["agreement"]:.1%}, Cohen kappa: {consistency["cohen_kappa"]:.3f}. 동일 모델 반복 안정성이며 사람 검증이 아님.','',
      '질의·판정에 D/T/M을 보여주지 않았고, 판정에 검색 방식·순위·인용을 보여주지 않았다. 모든 방식의 전체 코퍼스 상위 20개 합집합과 무작위 5개에 라벨을 부여했다. 소재 논문은 추천에서 제외했다. 하이퍼파라미터는 라벨 생성 전 고정했다.',
      '', 'LLM이 생성한 질의와 판정은 사람 정답 데이터가 아니다. Recall은 판정 풀 내 알려진 양성 기준이며 전체 코퍼스 Recall이 아니다. 그래프 방식은 PPR이며 1차 실험의 GNN 성능과 직접 연결하면 안 된다. 기존 결합 임베딩은 512토큰 제한에서 D/T/M이 잘렸을 수 있다. 현재 의도 분석 방식은 GPT6 Luna의 추가 질의 분석과 재정렬을 포함하므로 D/T/M만의 효과로 해석할 수 없다.']
    text+=['','## 해석','',
        'D/T/M 전용 표현은 제목·초록 기준선보다 NDCG@10이 낮았다. 기존 제목·초록+D/T/M 결합도 이번 실행에서는 소폭 낮았다. 인용 PPR 및 인용·저자 PPR은 평균 NDCG@10이 조금 높았지만 기준선 대비 95% 구간이 0을 포함한다. 현재 의도 분석·재정렬 역시 기준선과의 차이가 명확하지 않다. 따라서 이 결과로 자연어 추천 성능 향상을 입증했다고 주장할 수 없다.',
        '', '1차 인용 예측 실험의 향상은 해당 과제에서의 유용성을 뒷받침한다. 2차 실험은 과제·정답·기준선과 추천 방식이 다르므로 1차 실험과 직접적인 모순은 아니다. 자연어 추천에서는 현재의 D/T/M 입력·결합·재정렬 방식이 아직 효과를 입증하지 못한 것으로 해석한다.',
        '', '동일 판정 모델의 반복 일치율은 82.4%, Cohen kappa는 약 0.599이다. 일치율은 사람 정확도를 뜻하지 않으며, 특히 1점과 2점의 경계 및 판정 근거를 사람이 검토한 후 확증 평가를 하는 것이 필요하다. 소폭 점수 차이는 판정 변동성을 고려해 신중하게 해석한다.']
    (DATA/'RESULTS_KO.md').write_text('\n'.join(text),encoding='utf-8')
    print(OUT)

if __name__=='__main__':
    main()
