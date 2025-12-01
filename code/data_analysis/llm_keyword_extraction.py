import asyncio
import json
import os
import sys
import pandas as pd
import torch
from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import random

# --- 1. 환경 및 설정 ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY: sys.exit("오류: API 키 없음")
client = AsyncOpenAI(api_key=API_KEY)

# --- 설정 ---
TEST_MODE = False        
TEST_SIZE = 5           
OVERWRITE = False        
MAX_CONCURRENT_REQUESTS = 5   # Tier 1 최적화 (5개)
SAVE_INTERVAL = 50       

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
FFS_LOAD_FILE = os.path.join(project_root, 'subdataset', 'ogbn_arxiv_16k_ffs_sample.pt')
MAP_FILE = os.path.join(project_root, 'dataset', 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv')
TEXT_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv')
OUTPUT_DIR = os.path.join(project_root, 'output')
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'llm_extraction_results.json')
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, 'llm_error_log.json')

# --- 2. 데이터 로드 ---
print("1. 데이터 로드 중...")
_real_torch_load = torch.load
def _torch_load_with_weights_only_false(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = _torch_load_with_weights_only_false 

if not os.path.exists(FFS_LOAD_FILE): sys.exit("샘플 파일 없음")
loaded_data = torch.load(FFS_LOAD_FILE)
target_indices = set(loaded_data['indices'])

if not os.path.exists(MAP_FILE): sys.exit("매핑 파일 없음")
df_map = pd.read_csv(MAP_FILE)
df_map.columns = [c.strip().lower() for c in df_map.columns]
if 'node idx' in df_map.columns: df_map.rename(columns={'node idx': 'idx'}, inplace=True)
if 'idx' not in df_map.columns: df_map.rename(columns={df_map.columns[0]: 'idx', df_map.columns[1]: 'paper id'}, inplace=True)
df_target_map = df_map[df_map['idx'].isin(target_indices)].copy()
node_to_paper = dict(zip(df_target_map['idx'], df_target_map['paper id'].astype(str)))

print("   - 텍스트 로딩...")
try:
    df_text = pd.read_csv(TEXT_FILE, sep='\t', header=None, names=['paper_id', 'title', 'abstract'], dtype={'paper_id': str})
    target_paper_ids = set(node_to_paper.values())
    df_text = df_text[df_text['paper_id'].isin(target_paper_ids)].copy()
    paper_to_text = {}
    for row in df_text.itertuples():
        paper_to_text[str(row.paper_id)] = f"Title: {str(row.title)}\nAbstract: {str(row.abstract)}"
except Exception as e: sys.exit(f"텍스트 로드 실패: {e}")

processed_indices = set()
results_cache = []
error_cache = []

if not OVERWRITE and os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            processed_indices = {item['node_idx'] for item in existing_data}
            results_cache = existing_data
            print(f"기존 {len(processed_indices)}개 완료. 이어서 진행합니다.")
    except: pass

work_items = []
for idx in target_indices:
    if idx in processed_indices: continue
    pid = str(node_to_paper.get(idx, ""))
    text = paper_to_text.get(pid, "")
    if text: work_items.append({"idx": idx, "text": text})

if TEST_MODE: work_items = work_items[:TEST_SIZE]
print(f"처리할 작업: {len(work_items)}개")


# --- 3. Async LLM Processor (수다쟁이 모드) ---

async def process_single_item(sem, item):
    async with sem:
        node_idx = item['idx']
        text = item['text']
        
        system_prompt = "You are an expert Research Analyst. Extract core keywords."
        user_prompt = f"""
        Extract exactly **5 most critical keywords**. No reasoning. Output ONLY JSON.
        Paper Content: \"\"\"{text[:2000]}\"\"\"
        Output Format: {{ "features": ["kw1", "kw2", "kw3", "kw4", "kw5"] }}
        """

        max_retries = 10 # 재시도 횟수 늘림 (Tier 1은 오래 버텨야 함)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                await asyncio.sleep(0.2)
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2, max_tokens=100
                )
                
                content = response.choices[0].message.content
                parsed = json.loads(content)
                
                # 성공 시 1초 대기 (Rate Limit 예방)
                await asyncio.sleep(1.0)
                
                return {"node_idx": node_idx, "features": parsed.get("features", []), "status": "success"}
            
            except RateLimitError:
                # [수정됨] 멈추면 로그를 출력합니다!
                wait_time = 10 + (retry_count * 5) # 10초, 15초, 20초...
                tqdm.write(f"[일시정지] Node {node_idx}: 429 Rate Limit 발생! {wait_time}초간 대기합니다... (재시도 {retry_count+1}/{max_retries})")
                
                # 대기하는 동안 5초마다 생존 신고
                for i in range(wait_time, 0, -5):
                    if i > 5: # 너무 자주 뜨면 지저분하니까 5초 이상 남았을 때만
                        await asyncio.sleep(5)
                        # tqdm.write(f"   ...Node {node_idx} 대기 중 ({i-5}초 남음)")
                    else:
                        await asyncio.sleep(i)

                retry_count += 1
                
            except Exception as e:
                tqdm.write(f"[오류] Node {node_idx}: {e}")
                if "insufficient_quota" in str(e): return {"node_idx": node_idx, "status": "fatal"}
                await asyncio.sleep(2)
                retry_count += 1

        tqdm.write(f"[최종실패] Node {node_idx}: 재시도 횟수 초과")
        return {"node_idx": node_idx, "error": "Max retries exceeded", "status": "failed"}

async def main():
    if not work_items:
        print("작업 완료!")
        return

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_item(sem, item) for item in work_items]
    
    print(f"로그 출력 모드 시작 (동시 {MAX_CONCURRENT_REQUESTS}개)...")
    
    batch_results = []
    
    for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await f
        
        if result['status'] == 'fatal':
            print("치명적 오류로 중단합니다.")
            break
            
        if result['status'] == 'success':
            results_cache.append(result)
            batch_results.append(result)
        else:
            error_cache.append(result)
        
        if len(batch_results) >= SAVE_INTERVAL:
            save_checkpoint()
            batch_results = [] 
            
    if batch_results: save_checkpoint()
    if error_cache:
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(error_cache, f, ensure_ascii=False, indent=4)

    print("-" * 50)
    print(f"완료! 총 {len(results_cache)}개 저장됨.")
    print(f"파일: {OUTPUT_FILE}")

def save_checkpoint():
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_cache, f, ensure_ascii=False, indent=4)
    except: pass

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt:
        print("\n중단! 저장 중...")
        save_checkpoint()