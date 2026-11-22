"""Conservative bilingual query rules and offline condition evidence; no models."""
import re
from dataclasses import dataclass,asdict
import numpy as np
from dtm_graph_search import words

KOREAN={
 '라벨이 있는 노드가 적을 때':'few labeled nodes', '라벨이 적은':'few labels',
 '주변 연결 정보':'graph neighborhoods','노드 속성':'node attributes','노드 특성':'node features',
 '반지도':'semi supervised','노드 분류':'node classification','그래프 신경망':'graph neural network',
 '신경망 추론':'neural network inference','신경망':'neural network',
 '전력 소모':'power consumption','전력 소비':'power consumption','처리량':'throughput',
 '실제 측정':'measured evaluation','구현 논문':'implementation','구현':'implementation',
 '기관별':'institution','원본 데이터':'raw data','공동 학습':'collaborative learning',
 '연합 학습':'federated learning','차등 프라이버시':'differential privacy','보장':'guarantee',
 '비볼록 목적함수':'nonconvex objectives','비볼록':'nonconvex',
 '확률적 경사 하강법':'stochastic gradient descent','수렴':'convergence','증명':'proof',
 '명시적 평가 없이':'implicit feedback','암묵적 사용자 행동':'implicit feedback',
 '암묵적 피드백':'implicit feedback','클릭':'click','구매':'purchase','추천':'recommendation',
 '별점':'rating','서베이':'survey','리뷰':'review','설문':'survey','공격 논문':'attack','공격':'attack',
}


@dataclass
class QueryPlan:
    original:str
    positive_query:str
    exclusions:list
    facets:list
    evidence_requests:list
    warnings:list

    def to_dict(self):return asdict(self)


def parse_query(query):
    positive=[];negative=[]
    for clause in re.split(r'[.!?。;\n]+',query):
        if not clause.strip():continue
        # Preserve positive clauses; remove only explicit negative tails.
        match=re.search(r'\b(?:rather than|excluding|exclude|without\s+(?:surveys?|reviews?|attacks?)|not\s+(?:(?:a|an|the)|(?:surveys?|reviews?|attacks?|convex|strongly)))\b',clause,re.I)
        if match:
            positive.append(clause[:match.start()]);negative.append(clause[match.start():]);continue
        if re.search(r'제외|빼\s*줘|빼고',clause):
            recognized=list(re.finditer(r'(?:서베이|리뷰|공격(?:\s*논문)?)(?:은|는|을|를)?\s*(?:제외(?:해줘|하고|한|해)?|빼고|빼\s*줘)',clause))
            if recognized:
                negative.extend(m.group(0) for m in recognized)
                remainder=clause
                for m in reversed(recognized):remainder=remainder[:m.start()]+' '+remainder[m.end():]
                positive.append(remainder)
                continue
            parts=re.split(r'[,，]',clause)
            for p in parts:
                if re.search(r'제외|빼\s*줘|빼고',p):negative.append(p)
                else:positive.append(p)
        else:positive.append(clause)
    def translate(text):
        for k,v in sorted(KOREAN.items(),key=lambda x:-len(x[0])):text=text.replace(k,' '+v+' ')
        return text
    text=translate(' '.join(positive)).casefold();neg=translate(' '.join(negative)).casefold()
    text=re.sub(r'\b(?:without|no) explicit (?:ratings?|feedback)\b','implicit feedback',text)
    exclusions=[]
    if re.search(r'\bsurveys?\b|\breviews?\b',neg):exclusions.append('survey')
    if re.search(r'\battacks?\b',neg):exclusions.append('attack')
    if re.search(r'\bconvex\b',neg) or '볼록' in neg:exclusions.append('convex_only')
    cleaned=' '.join(words(text))
    facets=[]
    def add(name,pattern):
        if re.search(pattern,cleaned):facets.append(name)
    add('node_classification',r'node (?:classification|labeling)')
    add('fpga',r'\bfpga\b|field programmable gate array')
    add('neural_network',r'neural network|deep learning|\bdnn\b')
    add('differential_privacy',r'differential privacy|differentially private')
    add('collaborative_learning',r'collaborative learning|federated learning')
    add('nonconvex',r'\bnonconvex\b')
    add('stochastic_gradient',r'stochastic gradient')
    add('implicit_feedback',r'implicit (?:feedback|user item interaction)|absence explicit rating')
    evidence=[]
    if re.search(r'power consumption|power evaluation|energy consumption',cleaned):evidence.append('power')
    if 'throughput' in cleaned:evidence.append('throughput')
    if re.search(r'node (?:attribute|feature)',cleaned):evidence.append('node_features')
    warnings=[]
    if negative and not exclusions:warnings.append('Unrecognized exclusion clause; not applied automatically.')
    if re.search(r'[가-힣]',query):warnings.append('Korean uses a limited domain glossary, not general translation.')
    return QueryPlan(query,cleaned,exclusions,facets,evidence,warnings)


def build_metadata(papers,include_text=False):
    result=[]
    for p in papers:
        title=str(p.get('title','')).casefold();abstract=str(p.get('abstract','')).casefold();k=p.get('knowledge') or {};tags=[]
        for f in ('domain','task','method'):
            for source in (k,k.get('normalized',{}),k.get('graph',{})):
                if isinstance(source,dict):tags.extend(source.get(f,[]))
        dtm=' '+' '.join(words(' '.join(tags)))+' '
        facets=[]
        patterns={
          'node_classification':r'node classification|node labeling',
          'fpga':r'\bfpga\b|field programmable gate array',
          'neural_network':r'neural network|deep learning|\bdnn\b',
          'differential_privacy':r'differential privacy|differentially private',
          'collaborative_learning':r'collaborative (?:learning|machine learning|deep learning)|federated (?:learning|transfer learning)|multi party (?:learning|machine learning)',
          'nonconvex':r'\bnonconvex\b',
          'stochastic_gradient':r'stochastic gradient',
          'implicit_feedback':r'implicit feedback|implicit recommendation',
        }
        for f,pattern in patterns.items():
            if re.search(pattern,dtm):facets.append(f)
        text_facets=[]
        if include_text:
            text=' '+' '.join(words(title+' '+abstract))+' '
            text_facets=[f for f,pattern in patterns.items() if re.search(pattern,text)]
        patterns={'power':r'\b\d+(?:\.\d+)?\s*(?:mw|w|watts?)\b','throughput':r'\b(?:tops|gops|frames?/s|images?/s|fps)\b|throughput','node_features':r'node (?:feature|attribute)|text attribute'}
        evidence={};snippets={}
        for name,pattern in patterns.items():
            match=re.search(pattern,abstract);evidence[name]=bool(match)
            snippets[name]=str(p.get('abstract',''))[max(0,match.start()-100):min(len(abstract),match.end()+160)] if match else None
        result.append(dict(facets=facets,text_facets=text_facets,survey=bool(re.search(r'\bsurvey\b|\breview\b|systematic literature',title)),attack=bool(re.search(r'\battacks?\b|membership inference|information leakage',title)),convex_only='convex ' in dtm and 'nonconvex ' not in dtm,evidence=evidence,evidence_snippets=snippets))
    return result


class GuardedSearch:
    """Keep model and graph ranking behind explicit DTM topic eligibility."""
    def __init__(self,papers,extractor,model,metadata=None,use_text_bm25=True,consensus_strength=0.,min_routes=1):
        self.papers=papers;self.extractor=extractor;self.model=model
        self.consensus_strength=consensus_strength;self.min_routes=min_routes
        self.use_text_bm25=use_text_bm25
        self.metadata=metadata if metadata is not None else build_metadata(papers,include_text=use_text_bm25)
        if use_text_bm25:
            if metadata is not None and any('text_facets' not in m for m in metadata):raise ValueError('Text-aware metadata required')
            from text_bm25 import TextBM25
            self.text_index=TextBM25(papers)

    def search(self,query,top_k=10,excluded=None,candidate_pool=None):
        plan=parse_query(query);n=len(self.papers);eligible=np.ones(n,dtype=bool)
        for i,m in enumerate(self.metadata):
            supported=set(m['facets'])|set(m.get('text_facets',[]) if self.use_text_bm25 else [])
            if not set(plan.facets)<=supported:eligible[i]=False
            if 'survey' in plan.exclusions and m['survey']:eligible[i]=False
            if 'attack' in plan.exclusions and m['attack']:eligible[i]=False
            if 'convex_only' in plan.exclusions and m['convex_only'] and 'nonconvex' not in supported:eligible[i]=False
        if excluded is not None:eligible[excluded]=False
        # Apply topic and exclusion eligibility before seed selection, not just
        # after ranking; otherwise excluded surveys can dominate diffusion.
        state=self.extractor.prepare(plan.positive_query,excluded=excluded,lexical_mask=eligible)
        lexical=state['lex'];mass=state['mass']
        text_score=self.text_index.search(plan.positive_query) if self.use_text_bm25 else np.zeros(n)
        text_score[~eligible]=0
        available=np.flatnonzero(eligible)
        if not max(lexical.max(),text_score.max()) or not len(available):
            return dict(plan=plan.to_dict(),results=[],eligible_count=len(available),reason='No supported candidates for the recognized conditions; this does not prove corpus absence.')
        routes={}
        for name,values in (('dtm',lexical),('graph',mass),('text',text_score)):
            top=available[np.argsort(-values[available],kind='stable')[:200]]
            routes[name]={int(i):rank for rank,i in enumerate((i for i in top if values[i]>0),1)}
        if candidate_pool is None:
            candidates=sorted(set().union(*(set(v) for v in routes.values())))
        else:candidates=sorted(i for i in candidate_pool if eligible[i])
        if self.min_routes>1:candidates=[i for i in candidates if sum(i in r for r in routes.values())>=self.min_routes]
        if not candidates:return dict(plan=plan.to_dict(),results=[],eligible_count=len(available),reason='No eligible candidates in the supplied evaluation pool.')
        scores=self.model.predict(self.extractor.rows(state,candidates))
        fused=scores.copy()
        if self.use_text_bm25:
            # Rank fusion does not pretend the old model learned a new feature.
            order_model=np.argsort(-scores,kind='stable');model_rank=np.empty(len(candidates));model_rank[order_model]=np.arange(1,len(candidates)+1)
            fused=1/(60+model_rank)
            order_text=sorted((j for j,i in enumerate(candidates) if text_score[i]>0),key=lambda j:(-text_score[candidates[j]],candidates[j]))
            for rank,j in enumerate(order_text,1):fused[j]+=1/(60+rank)
        base_fused=fused.copy();agreement=[]
        for i in candidates:
            # Graph starts from DTM seeds: discount their correlated agreement.
            value=route_agreement({name:r[i] for name,r in routes.items() if i in r})
            agreement.append(value)
        fused+=self.consensus_strength*np.asarray(agreement)
        # Evidence is an ordering tier, never a claim that missing evidence disproves a condition.
        def key(j):
            i=candidates[j];supported=sum(self.metadata[i]['evidence'].get(e,False) for e in plan.evidence_requests)
            return (-supported,-float(fused[j]),i)
        order=sorted(range(len(candidates)),key=key)[:top_k]
        results=[dict(index=candidates[j],score=float(fused[j]),base_score=float(base_fused[j]),consensus_score=agreement[j],route_ranks={name:r[candidates[j]] for name,r in routes.items() if candidates[j] in r},route_count=sum(candidates[j] in r for r in routes.values()),model_score=float(scores[j]),text_bm25_score=float(text_score[candidates[j]]),conditions={e:'reported_in_abstract' if self.metadata[candidates[j]]['evidence'].get(e) else 'unknown' for e in plan.evidence_requests},evidence={e:self.metadata[candidates[j]]['evidence_snippets'].get(e) for e in plan.evidence_requests},matched_dtm_facets=self.metadata[candidates[j]]['facets'],matched_text_facets=self.metadata[candidates[j]].get('text_facets',[]) if self.use_text_bm25 else []) for j in order]
        return dict(plan=plan.to_dict(),results=results,eligible_count=len(available),reason=None)


def route_agreement(ranks):
    support={name:61/(60+rank) for name,rank in ranks.items()}
    return sum(w*support.get(a,0.)*support.get(b,0.) for a,b,w in (('dtm','text',1.),('text','graph',.5),('dtm','graph',.25)))/1.75
