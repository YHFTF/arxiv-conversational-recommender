import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import sys
import json
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
from torch.nn.functional import cosine_similarity # 코사인 유사도 사용을 위해 추가

# --- 0. 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY: 
    print("[Error] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    sys.exit(1)
client = OpenAI(api_key=API_KEY)
LLM_MODEL = "gpt-4o-mini"

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
LLM_RESULT_FILE = os.path.join(project_root, 'output', 'llm_extraction_results.json')
TITLE_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv')
INTERACTION_FILE = os.path.join(project_root, 'output', 'final_user_interactions.csv')
AUTHOR_DATA_FILE = os.path.join(project_root, 'output', 'author_data_openalex.json')

# 학습된 모델 파일 경로 (data1(llmv2).py에서 생성됨)
MODEL_LOAD_PATH = os.path.join(project_root, 'output', 'content_aware_net_v2.pth')

# 하이퍼파라미터 (학습 시 사용된 값과 동일해야 함)
EMBEDDING_DIM = 64
TOP_K = 5

# --- [확장성] 데이터 2/3 Reasoning 관련 설정 ---
INCLUDE_REASONING_FEATURE = False # Data 2/3 구축 완료 시 True로 변경
WEIGHT_DOMAIN = 2 # 논문 키워드 가중치
WEIGHT_TASK_METHOD = 1 # 논문 키워드 가중치
WEIGHT_REASONING = 1 # Reasoning 텍스트에 부여할 낮은 가중치

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] Device: {device}")


# --- 1. 모델 정의 (ContentAwareNet) ---
# 학습 시와 동일한 구조여야 합니다.
class ContentAwareNet(nn.Module):
    def __init__(self, num_users, num_items, num_keywords, embedding_dim):
        super(ContentAwareNet, self).__init__()
        
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.keyword_emb = nn.EmbeddingBag(num_keywords, embedding_dim, mode='mean', padding_idx=0)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        
    def forward(self, user, item, keyword_ids):
        # 이 함수는 학습 시에만 사용되지만, 구조 유지를 위해 보존
        u_vec = self.user_emb(user)            
        k_vec = self.keyword_emb(keyword_ids)  
        
        x = torch.cat([u_vec, k_vec], dim=1)   
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        logits = self.output(x)
        return logits.squeeze()

# --- 2. LLM을 사용하여 프롬프트에서 키워드 추출 (수정됨: 단일 리스트 요청) ---
def extract_structured_keywords_from_prompt(prompt, include_reasoning=False):
    """자연어 프롬프트에서 핵심 키워드 리스트를 추출합니다."""
    
    system_prompt = "You are an expert Research Analyst. Extract the core technical concepts from the user's research interest."
    
    reasoning_field = ""
    if include_reasoning:
        reasoning_field = ", \"reasoning\": \"...\""
        
    user_prompt = f"""
    Analyze the following research interest and extract 7 to 10 critical keywords. 
    Return ONLY a JSON object of the following format: {{ "keywords": [..] {reasoning_field} }}
    User Research Interest: "{prompt}"
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150
        )
        # LLM 출력은 {'keywords': [..]} 형태가 됩니다.
        return json.loads(response.choices[0].message.content) 
    except Exception as e:
        print(f"[Error] LLM API 호출 실패: {e}")
        return None

# --- 3. 프롬프트 키워드 -> 임베딩 벡터 생성 (단일 리스트 처리) ---
def create_prompt_keyword_tensor(structured_kws, keyword_to_id_map):
    """
    프롬프트 키워드를 모델 입력 텐서로 변환 (단순 키워드 리스트로 처리).
    """
    kw_ids = []
    
    # 1. [수정] Prompt Keywords 처리 (단일 리스트, 가중치 1배)
    # LLM 출력에서 'keywords' 필드를 사용합니다.
    for kw in structured_kws.get('keywords', []): 
        kw_lower = kw.strip().lower()
        if kw_lower in keyword_to_id_map:
            kw_ids.append(keyword_to_id_map[kw_lower]) # 1배 가중치 (default)

    # 2. [확장성] Reasoning 통합 (기존 로직 유지)
    if INCLUDE_REASONING_FEATURE:
        reasoning_text = structured_kws.get('reasoning', "")
        reasoning_words = [w.strip().lower() for w in reasoning_text.split() if w.isalpha()] 
        for word in reasoning_words:
            if word in keyword_to_id_map:
                kw_ids.extend([keyword_to_id_map[word]] * WEIGHT_REASONING)
                
    if not kw_ids: kw_ids = [0] 
    
    # 여기서 만들어지는 것은 '하나의' 프롬프트 키워드 인덱스 리스트입니다.
    return torch.tensor([kw_ids], dtype=torch.long).to(device)


# --- 4. 추천 엔진 (Cold-Start Mode - 코사인 유사도 최종 적용) ---
def get_recommendations_for_prompt(model, prompt_kw_tensor, all_item_kws_tensor, num_items, top_k):
    """
    Cold-Start 시 MLP의 편향을 피하고, 학습된 임베딩을 이용한 순수 코사인 유사도를 사용합니다.
    """
    num_total_items = num_items
    
    with torch.no_grad():
        # 1. Prompt Keyword Vector 생성 (V_prompt)
        # model.keyword_emb는 학습된 키워드 임베딩 가중치를 사용합니다.
        v_prompt = model.keyword_emb(prompt_kw_tensor) 

        # 2. Item Keyword Vector 생성 (V_item)
        v_item = model.keyword_emb(all_item_kws_tensor)
        
        # 3. [핵심] 코사인 유사도 계산
        # Prompt Vector를 Item 수만큼 복제하여 Item Vector와 코사인 유사도를 측정합니다.
        v_prompt_repeated = v_prompt.repeat(num_total_items, 1)
        
        # 코사인 유사도 측정 (출력 범위: -1.0 ~ 1.0)
        preds = cosine_similarity(v_prompt_repeated, v_item).cpu().numpy()
        
        # 스코어 정규화 (0.0% ~ 100.0%로 변환): (preds + 1) / 2
        # 음수 점수가 나오는 것을 방지하기 위해 0.0 ~ 1.0 범위로 변환 후 100을 곱합니다.
        preds = (preds + 1) / 2 
        
    top_indices = np.argsort(preds)[::-1][:top_k]

    return top_indices, preds


# --- 5. 데이터 로드 및 모델 로딩 (Item Keyword Tensor 로딩 로직 포함) ---
def load_all_data():
    if not os.path.exists(LLM_RESULT_FILE): sys.exit("[Error] LLM result file not found.")

    # 1. LLM 데이터 및 키워드 사전 로딩
    with open(LLM_RESULT_FILE, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    node_to_structured_keywords = {} 
    all_keywords = []
    
    for item in llm_data:
        node_idx = item['node_idx']
        
        # [수정] LLM 결과에서 'domain', 'task', 'method' 대신 'keywords'를 추출
        # 하지만 논문 쪽은 가중치 부여를 위해 기존 D/T/M 구조를 유지한다고 가정하고 로드합니다.
        # 기존 LLM 추출 파일이 D/T/M 구조라고 가정합니다.
        domain_kws = [k.strip().lower() for k in item.get('domain', [])]
        task_kws = [k.strip().lower() for k in item.get('task', [])]
        method_kws = [k.strip().lower() for k in item.get('method', [])]
        
        node_to_structured_keywords[node_idx] = {'domain': domain_kws, 'task': task_kws, 'method': method_kws}
        
        all_keywords.extend(domain_kws)
        all_keywords.extend(task_kws)
        all_keywords.extend(method_kws)
        
    keyword_counts = Counter(all_keywords)
    unique_keywords = sorted(keyword_counts.keys())
    keyword_to_id = {kw: i+1 for i, kw in enumerate(unique_keywords)}
    NUM_KEYWORDS = len(keyword_to_id) + 1

    # 2. 데이터 매핑 (모델 크기 산정용)
    df_full = pd.read_csv(INTERACTION_FILE)
    valid_nodes = set(node_to_structured_keywords.keys())
    df = df_full[df_full['item_id'].isin(valid_nodes)].copy()
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    num_users = len(user_ids)
    num_items = len(item_ids)

    reverse_item_map = {new: original for new, original in enumerate(item_ids)}

    # 3. 모든 아이템의 키워드 텐서 사전 계산 (학습 시의 가중치 로직 재현)
    item_keyword_indices = {}
    for new_id, original_id in reverse_item_map.items():
        structured_kws = node_to_structured_keywords.get(original_id, {'domain': [], 'task': [], 'method': []})
        kw_ids = []
        # 논문 키워드는 D/T/M 가중치를 그대로 사용합니다.
        for kw in structured_kws['domain']:
            if kw in keyword_to_id: kw_ids.extend([keyword_to_id[kw]] * WEIGHT_DOMAIN)
        for kw in structured_kws['task'] + structured_kws['method']:
            if kw in keyword_to_id: kw_ids.extend([keyword_to_id[kw]] * WEIGHT_TASK_METHOD)
        if not kw_ids: kw_ids = [0]
        item_keyword_indices[new_id] = kw_ids
    
    all_kws_list = [torch.tensor(item_keyword_indices[i], dtype=torch.long) for i in range(num_items)]
    all_item_kws_tensor = pad_sequence(all_kws_list, batch_first=True, padding_value=0).to(device)


    # 4. 모델 로드 및 초기화
    model = ContentAwareNet(num_users, num_items, NUM_KEYWORDS, EMBEDDING_DIM).to(device)
    if not os.path.exists(MODEL_LOAD_PATH): sys.exit(f"[Error] 모델 파일 {MODEL_LOAD_PATH}을 찾을 수 없습니다. 학습을 먼저 진행해주세요.")
    
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    
    # Cold-Start 추천을 위해 0번 유저 임베딩을 0벡터(중립)로 설정
    if model.user_emb.weight.data.size(0) > 0:
        model.user_emb.weight.data[0].fill_(0.0) 

    model.eval()

    # 5. 제목 및 저자 정보 로드
    df_titles = pd.read_csv(TITLE_FILE, sep='\t', header=None, usecols=[0, 1],
                            names=['paper_id', 'title'], dtype={'paper_id': str})
    paper_title_map = dict(zip(df_titles['paper_id'], df_titles['title']))
    
    try:
        with open(AUTHOR_DATA_FILE, 'r', encoding='utf-8') as f:
            author_raw = json.load(f)
        node_to_authors = {item['node_idx']: item['authors'] for item in author_raw}
    except: node_to_authors = {}

    # 최종 리턴값에 all_item_kws_tensor 추가
    return model, keyword_to_id, num_items, reverse_item_map, \
           node_to_structured_keywords, paper_title_map, node_to_authors, df_full, \
           all_item_kws_tensor


# --- 6. 메인 대화형 루프 ---
def main():
    print("="*60)
    print("    🚀 LLM 기반 대화형 논문 추천 시스템 (ContentAwareNet V2) 🚀")
    print("="*60)
    print("[System] 데이터 및 모델 로딩 중...")
    
    try:
        # load_all_data 호출부 수정 (all_item_kws_tensor 받기)
        model, keyword_to_id, num_items, reverse_item_map, \
        node_to_structured_keywords, paper_title_map, node_to_authors, df_full, \
        all_item_kws_tensor = load_all_data()
        
        print(f"[System] 로딩 완료. 총 논문 수: {num_items}, 총 키워드 수: {len(keyword_to_id)}")

    except Exception as e:
        print(f"[Fatal Error] 초기화 중 오류 발생: {e}")
        return

    print("\n[Start] 추천을 원하는 연구 분야를 설명해주세요. (Ctrl+C로 종료)")

    while True:
        try:
            prompt = input("\n>>> 당신의 연구 관심사/프롬프트: ")
            if not prompt.strip(): continue

            # 1. LLM을 사용하여 키워드 추출
            print("[System] 🧠 LLM이 프롬프트를 분석하여 키워드를 추출하는 중...")
            # LLM에게는 단일 키워드 리스트를 요청합니다.
            structured_kws = extract_structured_keywords_from_prompt(prompt, include_reasoning=INCLUDE_REASONING_FEATURE) 

            if structured_kws is None or not structured_kws.get('keywords'):
                print("[Warning] 유효한 키워드를 추출하지 못했습니다. 다시 시도해 주세요.")
                continue

            # 2. 추출된 키워드를 모델 입력 텐서로 변환
            prompt_kw_tensor = create_prompt_keyword_tensor(structured_kws, keyword_to_id)
            
            # 3. 추천 수행
            print("[System] ✨ ContentAwareNet V2가 추천을 계산하는 중...")
            top_indices, preds = get_recommendations_for_prompt(
                model, 
                prompt_kw_tensor, 
                all_item_kws_tensor, # Item Content 텐서 전달
                num_items,
                TOP_K
            )
            
            # 4. 결과 출력
            print("\n" + "="*60)
            print(f"** 분석된 키워드 (LLM) **")
            print(f"  - 추출된 키워드: {structured_kws.get('keywords', [])}")
            
            if 'reasoning' in structured_kws:
                print(f"  - Reasoning: {structured_kws['reasoning']}")
                
            print(f"\n[TOP {TOP_K} 추천 논문]")
            print("------------------------------------------------------------")
            
            for rank, i_new in enumerate(top_indices, 1):
                oid = reverse_item_map[i_new]
                kws_dict = node_to_structured_keywords.get(oid, {'domain': [], 'task': [], 'method': []})
                
                authors = node_to_authors.get(oid, ["Unknown Authors"])
                authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                
                try:
                    pid = df_full[df_full['item_id'] == oid].iloc[0]['paper_id']
                    title = paper_title_map.get(str(pid), "Title Not Found")
                except: title = "Unknown"
                
                score = preds[i_new]
                
                print(f"[{rank}] Score: {score:.1%}")
                print(f"Title:   {title}")
                print(f"Authors: {authors_str}")
                print(f"Domain:  {kws_dict['domain']}")
                print(f"Task:    {kws_dict['task']}")
                print(f"Method:  {kws_dict['method']}")
                print("------------------------------------------------------------")

        except KeyboardInterrupt:
            print("\n\n[System] Exiting by user request. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error] 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    # 코사인 유사도 사용을 위해 이 파일을 실행하기 전에 'from torch.nn.functional import cosine_similarity'를 추가해야 합니다.
    main()