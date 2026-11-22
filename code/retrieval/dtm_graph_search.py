"""CPU lexical D/T/M index; no language-model or embedding dependencies."""
from collections import Counter
import re
import unicodedata
import numpy as np
from scipy import sparse

STOP=set('a an the of for to in on and or with by from is are as at that this using use based without we our how methods method approach approaches studies study research paper papers'.split())
ALIASES={
    '그래프 신경망':'graph neural network','노드 분류':'node classification',
    '링크 예측':'link prediction','차등 프라이버시':'differential privacy',
    '연합 학습':'federated learning','추천 시스템':'recommender system',
    '확률적 경사 하강법':'stochastic gradient descent','비볼록':'nonconvex',
    'gnns':'graph neural network','gnn':'graph neural network',
    'cnns':'convolutional neural network','cnn':'convolutional neural network',
    'rnns':'recurrent neural network','rnn':'recurrent neural network',
    'nlp':'natural language processing','svm':'support vector machine',
    'sgd':'stochastic gradient descent','non-convex':'nonconvex',
    'non convex':'nonconvex','semi-supervised':'semisupervised',
    'semi supervised':'semisupervised','nearest-neighbor':'nearest neighbor',
}


def words(text):
    text=unicodedata.normalize('NFKC',str(text)).casefold()
    for term,value in sorted(ALIASES.items(),key=lambda kv:-len(kv[0])):
        text=re.sub(r'(?<!\w)'+re.escape(term)+r'(?!\w)',value,text)
    result=[]
    for w in re.findall(r'[a-z0-9]+',text):
        if len(w)>4 and w.endswith('ies'):w=w[:-3]+'y'
        elif len(w)>3 and w.endswith('s') and not w.endswith(('ss','us','is','ics')):w=w[:-1]
        if w not in STOP:result.append(w)
    return result


class DTMIndex:
    def __init__(self,papers):
        self.fields={f:[] for f in ('domain','task','method')}
        for p in papers:
            knowledge=p.get('knowledge') or {}
            for field in self.fields:
                tags=[]
                for source in (knowledge,knowledge.get('normalized',{}),knowledge.get('graph',{})):
                    if isinstance(source,dict):tags.extend(source.get(field,[]))
                seen=set();counter=Counter()
                for tag in tags:
                    ws=words(tag);key=tuple(ws)
                    if key in seen:continue
                    seen.add(key);counter.update('u:'+w for w in ws)
                    counter.update('b:'+a+' '+b for a,b in zip(ws,ws[1:]))
                self.fields[field].append(counter)
        self.vocab={t:i for i,t in enumerate(sorted({t for docs in self.fields.values() for doc in docs for t in doc}))}
        self.matrices={}
        for field,docs in self.fields.items():
            rows=[];cols=[];values=[];length=np.array([sum(v for t,v in d.items() if t.startswith('u:')) for d in docs],dtype=float)
            for i,d in enumerate(docs):
                for t,v in d.items():rows.append(i);cols.append(self.vocab[t]);values.append(v)
            tf=sparse.coo_matrix((values,(rows,cols)),shape=(len(docs),len(self.vocab)),dtype=float).tocsr()
            df=np.asarray((tf>0).sum(axis=0)).ravel();idf=np.log(1+(len(docs)-df+.5)/(df+.5));co=tf.tocoo()
            denom=co.data+1.2*(.25+.75*length[co.row]/max(length.mean(),1.))
            self.matrices[field]=sparse.coo_matrix((idf[co.col]*co.data*2.2/denom,(co.row,co.col)),shape=tf.shape).tocsr()

    def search(self,query,weights=(1.,1.,1.),phrase_weight=0.):
        ws=words(query);counts=Counter('u:'+w for w in ws)
        counts.update({'b:'+a+' '+b:phrase_weight for a,b in zip(ws,ws[1:])})
        q=np.zeros(len(self.vocab))
        for t,v in counts.items():
            if t in self.vocab:q[self.vocab[t]]=min(v,3.)
        score=sum(w*np.asarray(m@q).ravel() for w,m in zip(weights,self.matrices.values()))
        if score.max():score/=score.max()
        return score


def graph_scores(graph,lexical,seed_count=50,restart=.2,temperature=.05,excluded=None,centrality=None,debias=0.):
    score=lexical.copy()
    if excluded is not None:score[excluded]=-np.inf
    positive=np.flatnonzero(score>0)
    if not len(positive):return np.zeros_like(score)
    order=positive[np.argsort(-score[positive],kind='stable')[:seed_count]]
    values=np.exp((score[order]-score[order].max())/temperature);seed=np.zeros_like(score);seed[order]=values/values.sum()
    mass,_,ok=graph.diffuse(seed,True,False,restart=restart)
    if not ok:raise RuntimeError('PPR did not converge')
    if debias:
        if centrality is None:raise ValueError('Centrality is required')
        mass=mass/np.maximum(centrality,1e-12)**debias
    return mass


def rank_scores(scores,candidates,limit=None):
    """Corpus row is the deterministic tie break, independent of pool order."""
    return sorted(candidates,key=lambda i:(-float(scores[i]),int(i)))[:limit]
