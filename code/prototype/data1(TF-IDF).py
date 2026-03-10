import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import random
import json

# --- 1. 환경 설정 및 데이터 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 입력 파일 경로
INTERACTION_FILE = os.path.join(project_root, 'output', 'final_user_interactions.csv')
USER_PROFILE_FILE = os.path.join(project_root, 'output', 'final_user_profiles.json')
PAPER_KEYWORDS_FILE = os.path.join(project_root, 'output', 'paper_keywords_tfidf.json')
TITLE_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv') # 제목 로딩용

# 하이퍼파라미터 (Epoch 증가)
EMBEDDING_DIM = 64   # 차원 수도 조금 늘림 (표현력 증대)
BATCH_SIZE = 1024    # GPU 활용을 위해 배치 사이즈 키움
LEARNING_RATE = 0.005
EPOCHS = 30          # 학습 횟수 대폭 증가 (5 -> 30)
TOP_K = 5            

# 장치 설정 (CUDA 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 장치 설정: **{device}**")
if device.type == 'cuda':
    print(f"   - GPU Name: {torch.cuda.get_device_name(0)}")

print("\n1. 데이터 로드 및 전처리 중...")

# 1-1. 인터랙션 데이터
if not os.path.exists(INTERACTION_FILE):
    sys.exit(f"오류: {INTERACTION_FILE} 파일이 없습니다.")
df = pd.read_csv(INTERACTION_FILE, dtype={'paper_id': str})

# 1-2. 논문 제목 매핑 데이터 로드 (titleabs.tsv)
print("   - 논문 제목 데이터 로딩 중... (잠시만 기다려주세요)")
try:
    # TSV 로드 (헤더 없음: paper_id, title, abstract)
    df_titles = pd.read_csv(TITLE_FILE, sep='\t', header=None, usecols=[0, 1],
                            names=['paper_id', 'title'], dtype={'paper_id': str})
    # paper_id -> title 딕셔너리 생성
    paper_title_map = dict(zip(df_titles['paper_id'], df_titles['title']))
    print(f"   - 제목 매핑 완료: {len(paper_title_map):,}개 논문")
except Exception as e:
    print(f"제목 파일 로드 실패: {e}")
    paper_title_map = {}

# 1-3. 아이템 ID -> Paper ID 매핑 (추천 결과 해석용)
# item_id는 0부터 시작하는 인덱스, paper_id는 원본 ID
item_to_paper_id = df.set_index('item_id')['paper_id'].to_dict()

num_users = df['user_id'].max() + 1
num_items = df['item_id'].max() + 1
print(f"✅ 데이터 준비 완료: 유저 {num_users}명, 아이템 {num_items}개, 인터랙션 {len(df)}건")

# 키워드 로드
with open(PAPER_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
    paper_keywords = json.load(f)

# --- 2. 모델 정의 (Matrix Factorization) ---
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # Xavier Initialization (학습 초기화 개선)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        return (u * i).sum(1)

# --- 3. 데이터셋 정의 ---
class InteractionDataset(Dataset):
    def __init__(self, user_ids, item_ids, num_items, neg_ratio=4):
        self.users = user_ids
        self.items = item_ids
        self.num_items = num_items
        self.neg_ratio = neg_ratio
        self.users_set = set(zip(user_ids, item_ids))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        samples = [(u, i, 1.0)]
        for _ in range(self.neg_ratio):
            neg_item = random.randint(0, self.num_items - 1)
            while (u, neg_item) in self.users_set:
                neg_item = random.randint(0, self.num_items - 1)
            samples.append((u, neg_item, 0.0))
        return samples

# --- 4. 🏋️학습 (Training) ---
dataset = InteractionDataset(df['user_id'].values, df['item_id'].values, num_items)

def collate_fn(batch):
    users, items, labels = [], [], []
    for samples in batch:
        for u, i, l in samples:
            users.append(u)
            items.append(i)
            labels.append(l)
    # 텐서 생성 및 GPU 이동
    return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = MatrixFactorization(num_users, num_items, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n--- 학습 시작 (Epochs: {EPOCHS}) ---")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for user_batch, item_batch, label_batch in dataloader:
        # GPU로 이동
        user_batch = user_batch.to(device)
        item_batch = item_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(user_batch, item_batch)
        loss = criterion(predictions, label_batch.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 5 epoch마다 로그 출력
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# --- 5. 추천 및 결과 확인 ---
print("\n--- 추천 결과 시뮬레이션 ---")

model.eval()

# 헤비 유저 선정
heavy_users = df['user_id'].value_counts()
heavy_users = heavy_users[heavy_users >= 5].index.tolist()
if not heavy_users: heavy_users = df['user_id'].unique().tolist()
target_user_id = random.choice(heavy_users)

# 유저 정보 로드
with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
    profiles = json.load(f)
    user_info = next((p for p in profiles if p['user_id'] == target_user_id), None)
    user_name = user_info['author_name'] if user_info else "Unknown"

# 실제 히스토리
real_history_items = df[df['user_id'] == target_user_id]['item_id'].tolist()

# 예측 수행 (전체 아이템에 대해)
all_items = torch.arange(num_items).to(device)
target_user_tensor = torch.tensor([target_user_id] * num_items).to(device)

with torch.no_grad():
    predictions = model(target_user_tensor, all_items)
    scores = torch.sigmoid(predictions).cpu().numpy()

# 이미 본 것은 제외
scores[real_history_items] = -1
top_k_indices = np.argsort(scores)[::-1][:TOP_K]

# --- 6. 최종 리포트 출력 ---
print(f"\n**Target User**: {user_name} (ID: {target_user_id})")
print(f"**작성 논문 수**: {len(real_history_items)}편")
print(f"**주요 관심사 (Top Keywords)**: {user_info['top_keywords'] if user_info else 'N/A'}")

print(f"\n**[Top {TOP_K} 추천 논문]**")
print("-" * 60)

for rank, item_idx in enumerate(top_k_indices, 1):
    # 1. Paper ID 찾기
    paper_id = str(item_to_paper_id.get(item_idx, "Unknown"))
    
    # 2. 제목 찾기
    title = paper_title_map.get(paper_id, "Title Not Found")
    
    # 3. 키워드 찾기
    keywords = paper_keywords.get(paper_id, ["No Keywords"])
    
    # 4. 점수
    score = scores[item_idx]
    
    print(f"[{rank}위] (유사도: {score:.1%})")
    print(f"  **Title**: {title}")
    print(f"  **Keywords**: {', '.join(keywords)}")
    print("-" * 60)

print("\n추천 시스템 실행 완료.")