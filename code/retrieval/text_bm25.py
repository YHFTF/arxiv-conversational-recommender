"""CPU title/abstract BM25, with separate field length normalization."""
from collections import Counter
import numpy as np
from scipy import sparse
from dtm_graph_search import words


class TextBM25:
    def __init__(self,papers):
        docs={f:[Counter(words(str(p.get(f,'')))) for p in papers] for f in ('title','abstract')}
        self.vocab={t:i for i,t in enumerate(sorted({t for ds in docs.values() for d in ds for t in d}))}
        self.matrices={}
        for field,ds in docs.items():
            rows=[];cols=[];values=[];length=np.array([sum(d.values()) for d in ds],dtype=float)
            for i,d in enumerate(ds):
                for t,v in d.items():rows.append(i);cols.append(self.vocab[t]);values.append(v)
            tf=sparse.coo_matrix((values,(rows,cols)),shape=(len(papers),len(self.vocab)),dtype=float).tocsr()
            df=np.asarray((tf>0).sum(axis=0)).ravel();idf=np.log(1+(len(papers)-df+.5)/(df+.5));co=tf.tocoo()
            denom=co.data+1.2*(.25+.75*length[co.row]/max(float(length.mean()),1.))
            self.matrices[field]=sparse.coo_matrix((idf[co.col]*co.data*2.2/denom,(co.row,co.col)),shape=tf.shape).tocsr()

    def search(self,query):
        q=np.zeros(len(self.vocab))
        for t,count in Counter(words(query)).items():
            if t in self.vocab:q[self.vocab[t]]=min(count,3)
        score=3*np.asarray(self.matrices['title']@q).ravel()+np.asarray(self.matrices['abstract']@q).ravel()
        return score/max(float(score.max()),1e-12)

    @property
    def array_bytes(self):
        return sum(a.nbytes for m in self.matrices.values() for a in (m.data,m.indices,m.indptr))
