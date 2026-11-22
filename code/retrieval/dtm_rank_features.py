"""Interpretable lexical/graph scoring signals for a small CPU ranker."""
import numpy as np
from dtm_graph_search import words,graph_scores

FEATURES=['bm25','ppr','sqrt_ppr','centrality_corrected_ppr','domain_bm25','task_bm25','method_bm25','phrase_bm25','query_token_coverage','domain_coverage','task_coverage','method_coverage','best_tag_coverage','exact_tag_count','log_degree','log_centrality','neighbor_lexical_support']


class Features:
    def __init__(self,papers,legacy,index,graph):
        self.legacy=legacy;self.index=index;self.graph=graph
        self.tags=[];self.fields=[]
        for p in papers:
            fields={};tags=[];k=p.get('knowledge') or {}
            for f in ('domain','task','method'):
                values=[]
                for source in (k,k.get('normalized',{}),k.get('graph',{})):
                    if isinstance(source,dict):values.extend(source.get(f,[]))
                ws=[tuple(words(t)) for t in values];tags.extend(set(ws));fields[f]=set(w for t in ws for w in t)
            self.tags.append(tags);self.fields.append(fields)
        uniform=np.ones(len(papers))/len(papers);self.centrality,_,ok=graph.diffuse(uniform,True,False)
        if not ok:raise RuntimeError('Centrality failed')
        self.degree=np.diff(graph.citation.indptr)

    def prepare(self,query,excluded=None,lexical_mask=None):
        lex=self.legacy.search({'rewritten_query':query},False)[0]
        if lexical_mask is not None:
            if np.shape(lexical_mask)!=np.shape(lex):raise ValueError('Lexical mask shape mismatch')
            lex=np.where(lexical_mask,lex,0.)
        mass=graph_scores(self.graph,lex,excluded=excluded)
        norm=mass/max(mass.max(),1e-12)
        corr=mass/np.sqrt(np.maximum(self.centrality,1e-12));corr/=max(corr.max(),1e-12)
        typed=[self.index.search(query,weights=w) for w in ((1,0,0),(0,1,0),(0,0,1))]
        phrase=self.index.search(query,phrase_weight=1.)
        support=self.graph.citation@lex
        return dict(query=query,query_words=words(query),lex=lex,mass=mass,norm=norm,corr=corr,typed=typed,phrase=phrase,support=support)

    def rows(self,state,candidates):
        qw=state['query_words'];qset=set(qw);query=' '+' '.join(qw)+' ';rows=[]
        def coverage(doc):return len(qset&doc)/max(len(qset),1)
        for i in candidates:
            fields=self.fields[i];tags=self.tags[i];union=set().union(*fields.values())
            best=max((len(qset&set(t))/max(len(t),1) for t in tags),default=0.)
            exact=sum(bool(t) and ' '+' '.join(t)+' ' in query for t in tags)
            rows.append([state['lex'][i],state['norm'][i],np.sqrt(state['norm'][i]),state['corr'][i],*[s[i] for s in state['typed']],state['phrase'][i],coverage(union),*[coverage(fields[f]) for f in ('domain','task','method')],best,exact,np.log1p(self.degree[i]),np.log(max(self.centrality[i],1e-12)),state['support'][i]])
        return np.asarray(rows,dtype=np.float64).reshape(-1,len(FEATURES))
