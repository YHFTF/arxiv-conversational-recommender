import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import sys
import random
import json
from collections import Counter

# --- 1. 환경 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 파일 경로
INTERACTION_FILE = os.path.join(project_root, 'output', 'final_user_interactions.csv')
LLM_RESULT_FILE = os.path.join(project_root, 'output', 'llm_extraction_results.json')
TITLE_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv')
USER_PROFILE_FILE = os.path.join(project_root, 'output', 'final_user_profiles.json')
AUTHOR_DATA_FILE = os.path.join(project_root, 'output', 'author_data_openalex.json')
if not os.path.exists(AUTHOR_DATA_FILE):
    AUTHOR_DATA_FILE = os.path.join(project_root, 'output', 'author_data.json')
MODEL_SAVE_PATH = os.path.join(project_root, 'output', 'content_aware_net_v2.pth')

# 하이퍼파라미터 (규제 및 안정화 설정)
EMBEDDING_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.005  # MLP 구조를 위해 낮춤 (0.05 -> 0.005)
EPOCHS = 15            # 과적합 방지를 위해 15회로 제한
WEIGHT_DECAY = 1e-4    # L2 정규화 도입 (과적합 방지)
TOP_K = 5
# 키워드 가중치 설정 (Domain에 2배 가중치 부여)
WEIGHT_DOMAIN = 2
WEIGHT_TASK_METHOD = 1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] Device: {device}")

# --- 2. 데이터 로드 및 전처리 (수정됨: 구조화된 키워드 로드) ---
print("[System] Loading data...")

if not os.path.exists(LLM_RESULT_FILE): sys.exit("[Error] LLM result file not found.")

# 1. LLM 데이터 & 키워드 사전
with open(LLM_RESULT_FILE, 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

node_to_structured_keywords = {} # 구조화된 키워드를 저장
all_keywords = []
for item in llm_data:
    node_idx = item['node_idx']
    # 'domain', 'task', 'method' 필드를 모두 가져와 소문자로 변환 및 공백 제거
    domain_kws = [k.strip().lower() for k in item.get('domain', [])]
    task_kws = [k.strip().lower() for k in item.get('task', [])]
    method_kws = [k.strip().lower() for k in item.get('method', [])]
    
    node_to_structured_keywords[node_idx] = {
        'domain': domain_kws,
        'task': task_kws,
        'method': method_kws
    }
    
    all_keywords.extend(domain_kws)
    all_keywords.extend(task_kws)
    all_keywords.extend(method_kws)

keyword_counts = Counter(all_keywords)
unique_keywords = sorted(keyword_counts.keys())
keyword_to_id = {kw: i+1 for i, kw in enumerate(unique_keywords)}
NUM_KEYWORDS = len(keyword_to_id) + 1

# 2. 유저 프로필 로드 (기존과 동일)
try:
    with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
        user_profiles = json.load(f)
    user_id_to_name = {p['user_id']: p['author_name'] for p in user_profiles}
except:
    user_id_to_name = {}
    print("[Warning] Failed to load user profiles.")

# 3. 논문 저자 정보 로드 (기존과 동일)
try:
    with open(AUTHOR_DATA_FILE, 'r', encoding='utf-8') as f:
        author_raw = json.load(f)
    node_to_authors = {item['node_idx']: item['authors'] for item in author_raw}
except:
    node_to_authors = {}
    print("[Warning] Failed to load author data.")

# 4. 데이터 필터링 및 ID 재매핑 (기존과 동일)
df_full = pd.read_csv(INTERACTION_FILE)
valid_nodes = set(node_to_structured_keywords.keys()) # 키워드 딕셔너리 변경 반영
df = df_full[df_full['item_id'].isin(valid_nodes)].copy()

if len(df) < 10: sys.exit("[Error] Not enough data.")

user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

user_id_map = {original: new for new, original in enumerate(user_ids)}
item_id_map = {original: new for new, original in enumerate(item_ids)}
reverse_user_map = {new: original for new, original in enumerate(user_ids)}
reverse_item_map = {new: original for new, original in enumerate(item_ids)}

df['user_id_new'] = df['user_id'].map(user_id_map)
df['item_id_new'] = df['item_id'].map(item_id_map)

# 아이템별 키워드 인덱스 준비 (수정됨: 가중치 적용)
item_keyword_indices = {}
for new_id, original_id in reverse_item_map.items():
    structured_kws = node_to_structured_keywords.get(original_id, {'domain': [], 'task': [], 'method': []})
    
    kw_ids = []
    
    # 1. Domain (2배 가중치)
    for kw in structured_kws['domain']:
        if kw in keyword_to_id:
            kw_id = keyword_to_id[kw]
            kw_ids.extend([kw_id] * WEIGHT_DOMAIN) # 2번 반복

    # 2. Task (1배 가중치)
    for kw in structured_kws['task']:
        if kw in keyword_to_id:
            kw_ids.extend([keyword_to_id[kw]] * WEIGHT_TASK_METHOD) # 1번 반복

    # 3. Method (1배 가중치)
    for kw in structured_kws['method']:
        if kw in keyword_to_id:
            kw_ids.extend([keyword_to_id[kw]] * WEIGHT_TASK_METHOD) # 1번 반복

    if not kw_ids: kw_ids = [0]
    item_keyword_indices[new_id] = kw_ids

# 제목 로드 (기존과 동일)
try:
    df_titles = pd.read_csv(TITLE_FILE, sep='\t', header=None, usecols=[0, 1],
                            names=['paper_id', 'title'], dtype={'paper_id': str})
    paper_title_map = dict(zip(df_titles['paper_id'], df_titles['title']))
except: paper_title_map = {}

print(f"[System] Data prepared: {len(df)} interactions (Users: {len(user_ids)}, Papers: {len(item_ids)})")

# --- 3. 모델 정의 (수정됨: MLP 기반 ContentAwareNet) ---
class ContentAwareNet(nn.Module):
    def __init__(self, num_users, num_items, num_keywords, embedding_dim):
        super(ContentAwareNet, self).__init__()
        
        # 1. 임베딩 레이어
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.keyword_emb = nn.EmbeddingBag(num_keywords, embedding_dim, mode='mean', padding_idx=0)
        
        # 2. MLP 레이어 (Neural Network)
        # 입력 차원: User(64) + Keyword(64) = 128
        self.fc1 = nn.Linear(embedding_dim * 2, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1) 
        
        # 활성화 함수 및 드롭아웃 (과적합 방지)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        
        # 초기화
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.keyword_emb.weight)
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, user, item, keyword_ids):
        # item 인자는 호환성을 위해 받지만 사용하지 않음
        
        u_vec = self.user_emb(user)            # [Batch, Dim]
        k_vec = self.keyword_emb(keyword_ids)  # [Batch, Dim]
        
        # [핵심] 두 벡터를 이어 붙임 (Concatenate)
        x = torch.cat([u_vec, k_vec], dim=1)   # [Batch, Dim * 2]
        
        # MLP 통과
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        # 최종 예측 (Logits)
        logits = self.output(x)
        return logits.squeeze()

# --- 4. 데이터셋 (기존과 동일) ---
class HybridDataset(Dataset):
    def __init__(self, user_ids, item_ids, item_kw_map, num_items, neg_ratio=4):
        self.users = user_ids
        self.items = item_ids
        self.item_kw_map = item_kw_map
        self.num_items = num_items
        self.neg_ratio = neg_ratio
        self.users_set = set(zip(user_ids, item_ids))

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        u, i = self.users[idx], self.items[idx]
        pos_kws = torch.tensor(self.item_kw_map[i], dtype=torch.long)
        samples = [(u, i, pos_kws, 1.0)]
        for _ in range(self.neg_ratio):
            neg_item = random.randint(0, self.num_items - 1)
            while (u, neg_item) in self.users_set:
                neg_item = random.randint(0, self.num_items - 1)
            neg_kws = torch.tensor(self.item_kw_map[neg_item], dtype=torch.long)
            samples.append((u, neg_item, neg_kws, 0.0))
        return samples

def collate_fn(batch):
    users, items, keywords, labels = [], [], [], []
    for samples in batch:
        for u, i, k, l in samples:
            users.append(u)
            items.append(i)
            keywords.append(k)
            labels.append(l)
    keywords_padded = pad_sequence(keywords, batch_first=True, padding_value=0)
    return (torch.tensor(users), torch.tensor(items), keywords_padded, torch.tensor(labels))

# --- 5. 학습 및 모드 선택 (수정됨: ContentAwareNet 사용 및 규제 추가) ---
dataset = HybridDataset(df['user_id_new'].values, df['item_id_new'].values, item_keyword_indices, len(item_id_map))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# [수정] ContentAwareNet으로 모델 클래스 변경
model = ContentAwareNet(len(user_id_map), len(item_id_map), NUM_KEYWORDS, EMBEDDING_DIM).to(device)
# [수정] weight_decay 추가
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

print("\n" + "="*50)
print("   [실행 모드 선택]")
print("   1. 모델 새로 학습하기 (Train New Model)")
print("   2. 저장된 모델 불러오기 (Load Pre-trained Model)")
print("="*50)

mode = input(">> 번호를 입력하세요 (1 or 2): ").strip()

if mode == '1':
    # --- [Mode 1] 새로 학습 ---
    print(f"\n[Training] Started (Epochs: {EPOCHS})")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for u, i, k, l in dataloader:
            u, i, k, l = u.to(device), i.to(device), k.to(device), l.to(device)
            optimizer.zero_grad()
            loss = criterion(model(u, i, k), l.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Loss가 안정화된 후 (10회)부터 5회마다 출력
        if (epoch+1) % 5 == 0 or epoch == 0: 
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 학습 완료 후 모델 저장
    print(f"[System] Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("[System] Model saved successfully.")

elif mode == '2':
    # --- [Mode 2] 저장된 모델 로드 ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[System] Loading model from {MODEL_SAVE_PATH}...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("[System] Model loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            print("새로 학습을 진행해 주세요.")
            sys.exit()
    else:
        print(f"[Error] No saved model found at {MODEL_SAVE_PATH}.")
        print("먼저 '1번'을 선택하여 모델을 학습시켜 주세요.")
        sys.exit()

else:
    print("[Error] 잘못된 입력입니다. 프로그램을 종료합니다.")
    sys.exit()

# --- 6. 인터랙티브 랜덤 테스트 모드 (기존과 동일) ---
print("\n[System] Training complete.")
print("[System] Press 'Enter' to test a random user.")
print("[System] Press 'Ctrl+C' to exit.")

model.eval()

# 미리 계산된 전체 아이템 텐서 (추론 최적화)
all_items_new = torch.arange(len(item_id_map)).to(device)
all_kws_list = [torch.tensor(item_keyword_indices[i], dtype=torch.long) for i in range(len(item_id_map))]
all_kws_tensor = pad_sequence(all_kws_list, batch_first=True, padding_value=0).to(device)

# 랜덤 후보군 (논문 2편 이상 쓴 사람)
heavy_users = df['user_id_new'].value_counts()
candidates = heavy_users[heavy_users >= 2].index.tolist()
if not candidates: 
    print("[Error] No suitable candidates found.")
    sys.exit()

try:
    while True:
        input("\n>>> Press Enter to pick a random user... ")

        # 랜덤 유저 선택
        target_u_new = random.choice(candidates) # <--- 이 라인이 while 루프 안에 들여쓰기 되어 있어야 함
        target_u_original = reverse_user_map[target_u_new]
        target_u_name = user_id_to_name.get(target_u_original, f"Unknown User {target_u_original}")

        # 실제 히스토리
        real_items_new = df[df['user_id_new'] == target_u_new]['item_id_new'].tolist()

        # 추론
        all_user_tensor = torch.tensor([target_u_new] * len(item_id_map)).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(all_user_tensor, all_items_new, all_kws_tensor)).cpu().numpy()

        preds[real_items_new] = -1 
        
        # [수정된 부분]
        # Top 5 추천 논문 인덱스 (높은 점수 순)
        top_indices = np.argsort(preds)[::-1][:TOP_K] 
        
        # Bottom 5 비추천 논문 인덱스 (낮은 점수 순, -1 제외)
        # 이미 본 논문은 -1로 설정했으므로, 오름차순 정렬 시 -1을 건너뛴 다음 5개를 선택합니다.
        num_items_in_history = len(real_items_new)
        bottom_indices = np.argsort(preds)[num_items_in_history : num_items_in_history + TOP_K]

        # 결과 출력 (상단부)
        print("------------------------------------------------------------")
        print(f"[Target User] {target_u_name} (ID: {target_u_original})")
        
        print("[Actual Interests / Paper Keywords]")
        user_keywords_pool = []
        for i_new in real_items_new:
            oid = reverse_item_map[i_new]
            # 구조화된 키워드 출력 (새로운 딕셔너리 사용)
            kws_dict = node_to_structured_keywords.get(oid, {'domain': [], 'task': [], 'method': []})
            kws = kws_dict['domain'] + kws_dict['task'] + kws_dict['method']
            user_keywords_pool.extend(kws)
            print(f"  - Domain: {kws_dict['domain']}, Task: {kws_dict['task']}, Method: {kws_dict['method']}")


        # --- Top 5 추천 논문 출력 ---
        print(f"\n[TOP {TOP_K} 추천 논문 (높은 점수)]")
        print("------------------------------------------------------------")
        for rank, i_new in enumerate(top_indices, 1):
            oid = reverse_item_map[i_new]
            kws_dict = node_to_structured_keywords.get(oid, {'domain': [], 'task': [], 'method': []})
            
            # ... (이하 논문 정보 출력 로직은 기존과 동일) ...
            kws = kws_dict['domain'] + kws_dict['task'] + kws_dict['method']
            authors = node_to_authors.get(oid, ["Unknown Authors"])
            authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")

            try:
                pid = df_full[df_full['item_id'] == oid].iloc[0]['paper_id']
                title = paper_title_map.get(str(pid), "Title Not Found")
            except: title = "Unknown"
            
            score = preds[i_new]
            overlap = set(kws) & set(user_keywords_pool)
            
            print(f"[{rank}] Score: {score:.1%}")
            print(f"Title:   {title}")
            print(f"Authors: {authors_str}")
            print(f"Domain:  {kws_dict['domain']}")
            print(f"Task:    {kws_dict['task']}")
            print(f"Method:  {kws_dict['method']}")
            if overlap: print(f"Matches: {list(overlap)}")
            print("------------------------------------------------------------")


        # --- Bottom 5 비추천 논문 출력 (새로 추가) ---
        print(f"\n[BOTTOM {TOP_K} 비추천 논문 (낮은 점수)]")
        print("------------------------------------------------------------")
        for rank, i_new in enumerate(bottom_indices, 1):
            oid = reverse_item_map[i_new]
            kws_dict = node_to_structured_keywords.get(oid, {'domain': [], 'task': [], 'method': []})
            
            # ... (이하 논문 정보 출력 로직은 기존과 동일) ...
            kws = kws_dict['domain'] + kws_dict['task'] + kws_dict['method']
            authors = node_to_authors.get(oid, ["Unknown Authors"])
            authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")

            try:
                pid = df_full[df_full['item_id'] == oid].iloc[0]['paper_id']
                title = paper_title_map.get(str(pid), "Title Not Found")
            except: title = "Unknown"
            
            score = preds[i_new]
            overlap = set(kws) & set(user_keywords_pool)
            
            print(f"[{rank}] Score: {score:.1%}")
            print(f"Title:   {title}")
            print(f"Authors: {authors_str}")
            print(f"Domain:  {kws_dict['domain']}")
            print(f"Task:    {kws_dict['task']}")
            print(f"Method:  {kws_dict['method']}")
            if overlap: print(f"Matches: {list(overlap)}")
            print("------------------------------------------------------------")

except KeyboardInterrupt:
    print("\n\n[System] Exiting by user request. Goodbye!")
    sys.exit(0)