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
from llm_costing import LLMCostTracker

# --- 1. 환경 및 설정 ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("오류: API 키 없음")
client = AsyncOpenAI(api_key=API_KEY)

# run_costing.py 에서만 설정되는 코스트 측정 모드 플래그
LLM_COSTING_MODE = os.getenv("LLM_COSTING_MODE")

# 비용 산정용 단가 (gpt-4o-mini 기준, 필요 시 조정)
INPUT_TOKEN_COST_PER_MILLION = 0.15
OUTPUT_TOKEN_COST_PER_MILLION = 0.60

# --- 기본 설정 ---
TEST_MODE = False      
TEST_SIZE = 5           
OVERWRITE = False        
SAVE_INTERVAL = 50       

# [변경] 전역 설정 변수 (실행 시 사용자 입력에 의해 결정됨)
MAX_CONCURRENT_REQUESTS = 1
DELAY_BEFORE_REQUEST = 0.0

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


# --- 3. Async LLM Processor ---

async def process_single_item(sem, item):
    async with sem:
        node_idx = item['idx']
        text = item['text']
        
        # [수정] 시스템 프롬프트: 분류 기준 명시
        system_prompt = "You are an expert Research Analyst. Extract the core technical concepts from the paper's content, strictly classifying them into Domain, Task, and Method."
        
        # [수정] 사용자 프롬프트: JSON 구조 및 키워드 배분 가이드 명시
        user_prompt = f"""
        Extract a total of **5 to 7 most critical keywords** and classify them into the following three categories. Do not invent new keywords.
        - **Domain (1-2 KWs)**: The main research field (e.g., Computer Vision, NLP, Distributed Systems).
        - **Task (2-3 KWs)**: The specific problem being solved (e.g., Image Segmentation, Dialogue Generation, Phase Retrieval).
        - **Method (2 KWs)**: The core technique or architecture used (e.g., Transformer, GAN, Reinforcement Learning).

        Output ONLY JSON. No reasoning.
        Paper Content: \"\"\"{text[:2000]}\"\"\"
        Output Format: {{ 
            "domain": ["kw_dom1", "kw_dom2"], 
            "task": ["kw_task1", "kw_task2"], 
            "method": ["kw_method1", "kw_method2"]
        }}
        """
        
        max_retries = 10 
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                await asyncio.sleep(0.2)
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=150,  # 키워드 수가 늘었으므로 max_tokens를 150으로 증가
                )
                
                content = response.choices[0].message.content
                parsed = json.loads(content)

                # 토큰 사용량 추출 (코스트 측정용)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                
                await asyncio.sleep(1.0)
                
                # [수정] 결과 딕셔너리에 'features' 대신 'domain', 'task', 'method'를 포함
                #       코스트 측정을 위해 토큰 정보도 함께 반환
                return {
                    "node_idx": node_idx,
                    "domain": parsed.get("domain", []),
                    "task": parsed.get("task", []),
                    "method": parsed.get("method", []),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "status": "success",
                }
            
            except RateLimitError as e:  # [수정] 'as e'를 추가하여 에러 객체를 잡습니다.
                # 1. 429 에러의 구체적인 원인(메시지)을 추출하여 출력합니다.
                #    (이 메시지를 보면 RPM 초과인지, 하루 할당량(RPD) 초과인지 알 수 있습니다)
                error_detail = e.body.get('message', str(e)) if hasattr(e, 'body') and e.body else str(e)
                tqdm.write(f"\n[429 상세 원인] Node {node_idx}: {error_detail}")

                # 2. 대기 시간 설정 (보내주신 로직 유지)
                #    (주의: DELAY_BEFORE_REQUEST 변수가 코드 상단에 정의되어 있어야 에러가 안 납니다)
                wait_time = 3 if 'DELAY_BEFORE_REQUEST' in globals() and DELAY_BEFORE_REQUEST > 0 else (5 + retry_count * 2)
                
                tqdm.write(f"[일시정지] Node {node_idx}: 429 발생! {wait_time}초 대기...")
                await asyncio.sleep(wait_time)
                retry_count += 1
                
            except Exception as e:
                tqdm.write(f"[오류] Node {node_idx}: {e}")
                # 쿼터 부족(돈 없음/일일한도)은 치명적 오류로 처리
                if "insufficient_quota" in str(e): 
                    return {"node_idx": node_idx, "status": "fatal"}
                await asyncio.sleep(2)
                retry_count += 1

        tqdm.write(f"[최종실패] Node {node_idx}: 재시도 횟수 초과")
        return {"node_idx": node_idx, "error": "Max retries exceeded", "status": "failed"}

async def main():
    if not work_items:
        print("작업 완료!")
        return

    # 코스트 측정 모드일 때만 비용 트래커 활성화
    cost_tracker = None
    if LLM_COSTING_MODE:
        cost_tracker = LLMCostTracker(
            project_size=len(work_items),
            input_cost_per_million=INPUT_TOKEN_COST_PER_MILLION,
            output_cost_per_million=OUTPUT_TOKEN_COST_PER_MILLION,
        )

    # [설정 적용] 동시 요청 수 제한
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_item(sem, item) for item in work_items]
    
    print(f"▶ 실행 모드: 동시 {MAX_CONCURRENT_REQUESTS}개 요청 / 요청 간 대기 {DELAY_BEFORE_REQUEST}초")
    
    batch_results = []
    
    for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await f
        
        if result["status"] == "fatal":
            print("치명적 오류로 중단합니다.")
            break
            
        if result["status"] == "success":
            results_cache.append(result)
            batch_results.append(result)

            # 코스트 측정 모드일 때 토큰 사용량 기록
            if cost_tracker is not None:
                cost_tracker.add_call(
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    meta={"node_idx": result.get("node_idx")},
                )
        else:
            error_cache.append(result)
        
        if len(batch_results) >= SAVE_INTERVAL:
            save_checkpoint()
            batch_results = [] 
            
    if batch_results:
        save_checkpoint()
    if error_cache:
        with open(ERROR_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(error_cache, f, ensure_ascii=False, indent=4)

    # 코스트 측정 요약 출력
    if cost_tracker is not None:
        cost_tracker.finalize()
        cost_tracker.print_summary(label="Keyword Extraction Cost (full run)")

    print("-" * 50)
    print(f"완료! 총 {len(results_cache)}개 저장됨.")
    print(f"파일: {OUTPUT_FILE}")

def save_checkpoint():
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_cache, f, ensure_ascii=False, indent=4)
    except: pass

# --- 실행 진입점 (모드 선택) ---
if __name__ == "__main__":
    print("="*60)
    print(" [ OpenAI Tier 설정 ]")
    print(" 1. Tier 1 (안전 모드): 속도 하")
    print(" 2. Tier 2 (고속 모드): 속도 최상")
    print("="*60)
    
    while True:
        choice = input(">> 모드를 선택하세요 (1, 2): ").strip()
        if choice == '1':
            MAX_CONCURRENT_REQUESTS = 3     # 한번에 3개
            DELAY_BEFORE_REQUEST = 1.5      # 1.5초 대기
            print("\nTier 1 (안전 모드)")
            break
        elif choice == '2':
            MAX_CONCURRENT_REQUESTS = 50    # 한번에 50개
            DELAY_BEFORE_REQUEST = 0.0      # 대기 없음
            print("\nTier 2 (고속 모드)")
            break
        else:
            print("잘못된 입력입니다. 1, 2 중 하나를 입력해주세요.")

    try: asyncio.run(main())
    except KeyboardInterrupt:
        print("\n중단! 저장 중...")
        save_checkpoint()