import json
import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1.경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 입력 파일
INPUT_AUTHOR_FILE = os.path.join(project_root, 'output', 'author_data_openalex.json') # OpenAlex 우선
if not os.path.exists(INPUT_AUTHOR_FILE):
    INPUT_AUTHOR_FILE = os.path.join(project_root, 'output', 'author_data.json')

INPUT_TEXT_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv') # 텍스트 데이터

# 출력 파일
OUTPUT_DIR = os.path.join(project_root, 'output')
OUTPUT_INTERACTION_FILE = os.path.join(OUTPUT_DIR, 'final_user_interactions.csv')
OUTPUT_USER_PROFILE_FILE = os.path.join(OUTPUT_DIR, 'final_user_profiles.json')
OUTPUT_PAPER_KEYWORDS_FILE = os.path.join(OUTPUT_DIR, 'paper_keywords_tfidf.json')

# --- 2.데이터 로드 ---
print("1. 데이터 로드 중...")

# 2-1. 저자 데이터 로드
try:
    with open(INPUT_AUTHOR_FILE, 'r', encoding='utf-8') as f:
        author_data = json.load(f)
    print(f"저자 데이터 로드 완료: {len(author_data)}개 논문")
except Exception as e:
    print(f"저자 데이터 로드 실패: {e}")
    sys.exit(1)

# 2-2. 텍스트 데이터 로드 (titleabs.tsv)
# 저자 데이터에 있는 논문만 필터링하기 위해 ID 세트 생성
target_paper_ids = {str(item['paper_id']) for item in author_data}

print("   - TSV 파일 읽는 중... (시간이 조금 걸릴 수 있습니다)")
try:
    # 헤더가 없다고 가정하고 로드 (paper_id, title, abstract)
    df_text = pd.read_csv(INPUT_TEXT_FILE, sep='\t', header=None, 
                          names=['paper_id', 'title', 'abstract'], dtype={'paper_id': str})
    
    # 우리가 필요한 논문만 필터링
    df_target = df_text[df_text['paper_id'].isin(target_paper_ids)].copy()
    
    # 텍스트 합치기 (제목 + 초록) -> 키워드 추출용
    df_target['full_text'] = df_target['title'].fillna('') + " " + df_target['abstract'].fillna('')
    print(f"텍스트 데이터 매칭 완료: {len(df_target)}개")
    
except Exception as e:
    print(f"TSV 로드 실패: {e}")
    sys.exit(1)


# --- 3.핵심 키워드 3개 추출 (TF-IDF 사용) ---
print("\n2. 각 논문의 핵심 키워드 3개 추출 중 (TF-IDF)...")

# 전처리 함수
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # 영문과 공백만 남김
    return text

df_target['clean_text'] = df_target['full_text'].apply(clean_text)

# TF-IDF 설정 (불용어 제거, 상위 5000개 단어만 고려)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, max_df=0.95, min_df=2)
tfidf_matrix = vectorizer.fit_transform(df_target['clean_text'])
feature_names = np.array(vectorizer.get_feature_names_out())

# 상위 3개 키워드 뽑기
paper_keywords_map = {} # {paper_id: [kw1, kw2, kw3]}

for i, row in enumerate(df_target.itertuples()):
    # 해당 문서의 TF-IDF 벡터
    feature_index = tfidf_matrix[i, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    
    # 점수 높은 순 정렬
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    
    # 상위 3개 단어 추출
    top_3_indices = [idx for idx, score in sorted_scores[:3]]
    top_3_words = feature_names[top_3_indices].tolist()
    
    paper_keywords_map[str(row.paper_id)] = top_3_words

print(f"키워드 추출 완료. (예시: {list(paper_keywords_map.values())[0]})")

# 중간 저장 (키워드 데이터)
with open(OUTPUT_PAPER_KEYWORDS_FILE, 'w', encoding='utf-8') as f:
    json.dump(paper_keywords_map, f, indent=4)


# --- 4.유저 생성 (저자 + 키워드 결합) ---
print("\n3. 저자별 키워드 집계 및 유저 프로필 생성 중...")

user_registry = {} # { "Author Name": { "id": 0, "keywords": Counter(), "paper_ids": [] } }
user_id_counter = 0
interactions = []

for entry in author_data:
    pid = str(entry['paper_id'])
    authors = entry['authors']
    
    # 키워드가 있는 논문인가?
    if pid not in paper_keywords_map:
        continue
        
    keywords = paper_keywords_map[pid]
    
    for author in authors:
        name = author.strip()
        if not name: continue
        
        # 유저 등록
        if name not in user_registry:
            user_registry[name] = {
                "user_id": user_id_counter,
                "keyword_history": [],
                "paper_history": []
            }
            user_id_counter += 1
            
        # 데이터 누적
        user_registry[name]['keyword_history'].extend(keywords)
        user_registry[name]['paper_history'].append(pid)
        
        # 인터랙션 기록
        interactions.append({
            "user_id": user_registry[name]['user_id'],
            "item_id": entry['node_idx'], # 학습엔 Node Index 사용
            "paper_id": pid,
            "keywords": keywords # 이 인터랙션 당시의 관심사
        })

# --- 5.유저 프로필 정제 (Top Keywords 선정) ---
final_user_profiles = []

for name, data in user_registry.items():
    # 이 저자가 가장 많이 사용한 키워드 Top 5 추출
    from collections import Counter
    keyword_counts = Counter(data['keyword_history'])
    top_interest = [k for k, v in keyword_counts.most_common(5)]
    
    final_user_profiles.append({
        "user_id": data['user_id'],
        "author_name": name,
        "top_keywords": top_interest, # 유저의 대표 관심사
        "total_papers": len(data['paper_history'])
    })

# --- 6.저장 ---
print("-" * 50)
df_interactions = pd.DataFrame(interactions)
print(f"최종 유저 수: {len(final_user_profiles):,}명")
print(f"최종 인터랙션 수: {len(df_interactions):,}건")
print(f"   (유저당 평균 {len(df_interactions)/len(final_user_profiles):.2f}건 논문 작성)")

# 수정된 부분: 'paper_id' 컬럼을 포함하여 저장해야 합니다!
df_interactions[['user_id', 'item_id', 'paper_id', 'keywords']].to_csv(OUTPUT_INTERACTION_FILE, index=False)
print(f"인터랙션 파일: {OUTPUT_INTERACTION_FILE}")

# 2. 유저 프로필 저장
with open(OUTPUT_USER_PROFILE_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_user_profiles, f, ensure_ascii=False, indent=4)
print(f"유저 프로필: {OUTPUT_USER_PROFILE_FILE}")

print("\n[Step 1] 유저 생성 단계가 완료되었습니다.")