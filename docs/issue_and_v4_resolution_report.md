# 프로젝트 개선 보고서: V4 지식(Knowledge) 임베딩 연동 (Resolution Report)

## 📌 1. 기존 모델(V3/V2)의 구조적 한계점 (The Issue)
기존 `LGCmodel_v2.py` 및 이를 사용하는 V3 스크립트에는 논문의 메타데이터(Domain, Task, Method)를 다루기 위해 `nn.Embedding` 레이어들을 선언해 두었습니다.

```python
self.domain_emb = nn.Embedding(...)
self.task_emb = nn.Embedding(...)
self.method_emb = nn.Embedding(...)
```

하지만, 초기화(`__init__`) 과정에서만 선언되었을 뿐, 실제 신경망이 학습되는 순전파(`forward()`) 로직에서 이 임베딩들을 기존 논문 피처(`paper_x`)에 결합하는 과정이 누락되어 있었습니다. 

그 결과, BPR(Bayesian Personalized Ranking) Loss를 통해 오차를 계산하고 역전파(Backpropagation)를 수행할 때 지식 노드 쪽으로 파라미터 업데이트 텐서(Gradient)가 도달하지 못해 **"메모리에 선언은 되어 있으나 학습의 영향을 전혀 받지 못하는(가중치 업데이트가 없는)"** 문제가 있었습니다.

## 📌 2. 해결 방안: V4 Architecture 개선 (The Resolution)

이 현상을 해결하기 위해 `LGCmodel_v4.py`라는 새로운 신경망 구조를 구축했습니다.

### 2.1. 동적 지식 결합 (Dynamic Knowledge Injection)
모델에 새롭게 `_get_combined_embedding()` 메서드를 추가하여 백프로퍼게이션 파이프라인의 **단절 구간을 복원**했습니다.

```python
def _get_combined_embedding(self):
    # 1. 지식 Embedding 레이어 통과 (Gradient 추적이 시작되도록 함)
    d_val = self.domain_emb(self.paper_knowledge_ids[:, 0])
    t_val = self.task_emb(self.paper_knowledge_ids[:, 1])
    m_val = self.method_emb(self.paper_knowledge_ids[:, 2])
    
    # 2. 텍스트 피처와 지식 피처 합연산(Add)
    dynamic_paper_x = self.paper_base_x + d_val + t_val + m_val
    
    return torch.cat([dynamic_paper_x, self.author_emb, self.topic_emb], dim=0)
```

그래프 합성 전파(Message Passing)를 시작하기 전에, 지식 피처를 텍스트 피처와 더하여(Add) 전달하도록 수정했습니다. 이 조치를 통해 BPR 모델 오차가 역전파될 때 **Domain, Task, Method 임베딩 파라미터 쪽으로 미분값이 흘러 들어가 정상적인 학습을 수행**하게 됩니다.

### 2.2. Author & Topic 노드의 그래디언트 보호 (Parameter화) 
기존에는 Author와 Topic 노드의 피처를 `paper_x`의 평균값 단순 행렬 계산으로 만들고 고정시켰습니다. 이 또한 학습에서 제외되어 있었습니다. V4 모델에서는 이를 `nn.Parameter` 텐서로 감싸주어, 초기값은 기존처럼 논문의 평균으로 잡되 **학습 과정 진행 중 각 저자(Author)와 토픽(Topic) 노드가 모델 오차율을 줄이기 위해 최적의 벡터로 스스로 진화**하도록 변경했습니다.

## 📌 3. 파이프라인(스크립트) 업데이트 내역
기존 V2, V3 코드를 보존하고 안전하게 테스트하기 위해 아래 코드를 신규 생성했습니다.
1. **`train_v4_knowledge_bpr.py`**:
   - `arxiv_master_final.json`을 파싱하여 16,000편의 논문에 대한 `Domain, Task, Method` 정수형 ID 텐서 (`[16000, 3]`)를 즉석에서 생성하는 전처리 로직을 도입했습니다.
2. **`inference_v4.py`**:
   - 학습이 끝난 `lightgcn_v4_knowledge_bpr.pt` 가중치를 이용해 실제 논문을 추천해 보는 테스트 스크립트를 생성했습니다.

## 🚀 향후 기대 효과
이제 추천 모델이 오직 '인용(Citation)' 이라는 표면적인 관계 데이터만 모방하는 것을 넘어, **"이 논문은 Computer Vision 분야고 3D Object Detection 기법을 쓰니까 이와 관련된 논문들과도 더 유사할 것이다"**라는 복잡한 메타데이터 상관관계를 본격적으로 학습하게 되었습니다. V4 모델을 학습시키고 나면 이전 버전을 아득히 뛰어넘는 디테일한 추천 성능을 보실 수 있을 것입니다.
