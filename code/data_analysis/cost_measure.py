# cost_measure.py

import torch
import random
import sys
import os
import pandas as pd
import numpy as np
import json
import time


# --- 환경 설정 ---
FFS_LOAD_FILE = 'ogbn_arxiv_16k_ffs_sample.pt'
SAMPLE_FOR_COSTING = 100

# 🚨 필수 경로 설정
NODE_TO_ID_MAP_PATH = 'C:/Users/yungh/Desktop/data/dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv' 
TITLEABS_TSV_PATH = 'C:/Users/yungh/Desktop/data/titleabs.tsv'


# --- 1. 📂 저장된 FFS 샘플 파일 로드 ---
print("1. FFS 샘플 파일 로드 시작...")
# PyTorch 2.6+ 버전의 보안 문제를 우회하기 위한 패치 (loader와 동일)
_real_torch_load = torch.load
def _torch_load_with_weights_only_false(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = _torch_load_with_weights_only_false 

try:
    if not os.path.exists(FFS_LOAD_FILE):
        print(f"오류: '{FFS_LOAD_FILE}' 파일을 찾을 수 없습니다. main_sampler.py를 먼저 실행하세요.")
        sys.exit(1)
        
    loaded_data = torch.load(FFS_LOAD_FILE)
    sampled_node_list = loaded_data['indices']
    print(f"✅ FFS 샘플 데이터 로드 완료. 총 노드 수: {len(sampled_node_list)}개")

except Exception as e:
    print(f"⛔ 파일 로드 중 심각한 오류 발생: {e}")
    sys.exit(1)


# --- 2. 🔍 비용 산정용 100개 인덱스 추출 ---
cost_sample_indices = random.sample(sampled_node_list, SAMPLE_FOR_COSTING)
print(f"✅ 비용 산정용 {SAMPLE_FOR_COSTING}개 노드 인덱스 추출 완료.")


# --- 3. 🗺️ 노드 인덱스 -> 논문 ID 매핑 로드 ---
def load_node_to_paperid_map(map_path):
    # (이전에 제공된 맵핑 로드 로직 그대로 사용)
    try:
        mapping_df = pd.read_csv(map_path, header=0, dtype={'idx': np.int32, 'paper id': str})
        return mapping_df['paper id'].tolist()
    except Exception as e:
        print(f"⛔ 맵핑 파일 로드 오류: {e}")
        sys.exit(1)

node_id_list = load_node_to_paperid_map(NODE_TO_ID_MAP_PATH)
print("✅ 노드 인덱스-논문 ID 맵핑 로드 완료.")


# --- 4. 📝 초록 텍스트 추출 (TSV 사용) ---
def extract_abstracts_from_tsv(tsv_path, node_id_list, target_indices):
    # (이전에 제공된 TSV 텍스트 추출 로직 그대로 사용)
    target_paper_ids = {node_id_list[idx] for idx in target_indices}
    
    try:
        # TSV 파일 로드 (헤더 없음 가정)
        df_texts = pd.read_csv(tsv_path, sep='\t', header=None, 
                               names=['paper id', 'title', 'abstract'], 
                               dtype={'paper id': str})
    except Exception as e:
        print(f"⛔ TSV 파일 로드 오류. 경로, 구분자 또는 인코딩을 확인해 주세요: {e}")
        sys.exit(1)

    df_filtered = df_texts[df_texts['paper id'].isin(target_paper_ids)].set_index('paper id')

    final_abstracts = []
    for idx in target_indices:
        paper_id = node_id_list[idx]
        if paper_id in df_filtered.index:
            abstract = df_filtered.loc[paper_id, 'abstract']
            cleaned_abstract = abstract.replace("Abstract", "", 1).strip()
            final_abstracts.append(cleaned_abstract)
        else:
            final_abstracts.append("ERROR: Abstract not found in TSV.")

    return final_abstracts


abstract_texts = extract_abstracts_from_tsv(
    TITLEABS_TSV_PATH, 
    node_id_list, 
    cost_sample_indices
)

print("-" * 50)
print(f"✅ LLM 비용 산정용 **{len(abstract_texts)}개** 초록 텍스트 추출 완료.")
print("--- 샘플 텍스트 ---")
print(f"첫 번째 샘플 인덱스: {cost_sample_indices[0]}")
print(f"첫 번째 샘플 초록 (일부): {abstract_texts[0][:200]}...")
print("-" * 50)
print("🚀 이제 이 텍스트들을 LLM API에 입력하여 토큰 사용량을 측정할 수 있습니다.")


import openai # LLM API 호출을 위한 라이브러리
# import os # 이미 파일 상단에 있음

# --- 🚨 LLM API 설정 ---
# 1. API 키 설정 (실제 키로 대체해야 합니다)
os.environ["OPENAI_API_KEY"] = "api 키" 
client = openai.OpenAI()

# 2. 사용할 LLM 모델 및 단가 설정 (예시: GPT-3.5 Turbo)
LLM_MODEL = "gpt-4o-mini"
# (단가 예시: 1M 토큰당 Input $0.50, Output $1.50)
INPUT_TOKEN_COST_PER_MILLION = 0.15
OUTPUT_TOKEN_COST_PER_MILLION = 0.60

# --- 시뮬레이션 모드 (실제 API 호출 없이 토큰만 추정할 경우) ---
# 실제 API 호출을 원하지 않는 경우 이 변수를 True로 설정합니다.
SIMULATION_MODE = False

def create_llm_prompt(abstract):
    """LLM에 입력할 프롬프트를 생성합니다."""
    
    # 시스템 프롬프트: 모델의 역할과 목표를 정의합니다.
    system_prompt = (
        "You are a Thesis Analysis AI. Read the provided abstract and extract a maximum of 10 keywords with sufficient importance that best represent the core content of the thesis as 'Features'. Additionally, provide a concise rationale, under 50 words, for predicting the thesis's topic based on these extracted keywords."
    )
    
    # 사용자 입력: 분석할 초록 텍스트
    user_prompt = f"Thesis Abstract: \"\"\"{abstract}\"\"\""

    # 출력 형식 요청: JSON 형태로 받아 파싱하기 쉽게 합니다.
    format_prompt = (
        "The output must only be in the following JSON format: { \"features\": [\"word1\", \"word2\", ...], \"reasoning\": \"rationale text\" }"
    )

    return system_prompt + user_prompt + format_prompt


def call_llm_and_measure_tokens(abstract_text, model=LLM_MODEL, simulation=SIMULATION_MODE):
    """LLM을 호출하고 Input/Output 토큰을 측정합니다."""
    
    prompt = create_llm_prompt(abstract_text)
    
    if simulation:
        # 시뮬레이션 모드: 토큰 수를 임의로 추정합니다.
        # (실제 토큰 수를 얻으려면 API 호출 필요)
        input_tokens = len(prompt) // 4 + 50 # 대략적인 토큰 추정 (4글자당 1토큰 + 오차)
        output_tokens = random.randint(50, 100) # Output은 50~100 토큰으로 추정
        llm_output = {"features": ["simulated", "words"], "reasoning": "This is a simulated reasoning for cost measurement."}
        
    else:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.split('\n')[0]}, # 시스템 프롬프트 분리
                    {"role": "user", "content": "\n".join(prompt.split('\n')[1:])}
                ],
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            # LLM 출력은 JSON string이므로 파싱해야 합니다.
            llm_output = json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"LLM API 호출 오류 발생: {e}")
            input_tokens, output_tokens, llm_output = 0, 0, None

    return input_tokens, output_tokens, llm_output

# cost_measure.py의 가장 마지막 부분에 추가

# 100개 샘플에 대한 결과를 저장할 리스트
llm_results = []
total_input_tokens = 0
total_output_tokens = 0

print("-" * 50)
print(f"3. LLM 호출 및 토큰 측정 시작 (샘플 수: {len(abstract_texts)}개, 모델: {LLM_MODEL})")

total_start_time = time.perf_counter()
total_llm_call_duration = 0


for i, abstract_text in enumerate(abstract_texts):
    
    if "ERROR" in abstract_text:
        print(f"경고: 인덱스 {cost_sample_indices[i]} 텍스트 누락. 스킵합니다.")
        continue
    print(f"처리 중: {i + 1}/{len(abstract_texts)}번째 샘플 (인덱스: {cost_sample_indices[i]})")

    call_start_time = time.perf_counter()
    input_t, output_t, output_data = call_llm_and_measure_tokens(abstract_text)
    call_end_time = time.perf_counter()
    call_duration = call_end_time - call_start_time # 해당 호출 시간
    total_llm_call_duration += call_duration # 총 시간에 누적
    
    total_input_tokens += input_t
    total_output_tokens += output_t

    llm_results.append({
        'node_index': cost_sample_indices[i],
        'features': output_data['features'] if output_data else None,
        'reasoning': output_data['reasoning'] if output_data else None,
        'input_tokens': input_t,
        'output_tokens': output_t
    })
total_end_time = time.perf_counter()
overall_duration = total_end_time - total_start_time

print(f"총 LLM 호출 실행 시간 (LLM API Latency): **{total_llm_call_duration:.2f}초**")
print(f"전체 루프 실행 시간 (Total Runtime): **{overall_duration:.2f}초**")
print("-" * 50)
print("✅ LLM 호출 및 토큰 측정 완료.")

# --- 4. 💰 최종 비용 산정 및 예측 ---

if len(abstract_texts) > 0:
    # 100개 샘플의 평균 토큰 사용량
    num_processed = len(llm_results)
    avg_input_token = total_input_tokens / num_processed
    avg_output_token = total_output_tokens / num_processed

    # 1.6만 개로 확장하여 예상 토큰 사용량 계산
    PROJECT_SIZE = 16000
    projected_input_tokens = avg_input_token * PROJECT_SIZE
    projected_output_tokens = avg_output_token * PROJECT_SIZE

    # 예상 비용 계산 (단위: USD)
    projected_cost_input = (projected_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION
    projected_cost_output = (projected_output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_MILLION
    total_projected_cost = projected_cost_input + projected_cost_output

    print("\n--- 💰 LLM 비용 산정 결과 (1.6만 개 기준) ---")
    print(f"모델: {LLM_MODEL}")
    print(f"샘플당 평균 Input 토큰: {avg_input_token:.2f}")
    print(f"샘플당 평균 Output 토큰: {avg_output_token:.2f}")
    print(f"총 예상 Input 토큰 (16k): {projected_input_tokens:,.0f}개")
    print(f"총 예상 Output 토큰 (16k): {projected_output_tokens:,.0f}개")
    print(f"총 예상 LLM 비용: **${total_projected_cost:.2f} USD** (Input: ${projected_cost_input:.2f}, Output: ${projected_cost_output:.2f})")
    print("-" * 50)

# 추출된 Features와 Reasoning 예시 확인
print("--- 추출된 Feature/근거 예시 (첫 번째 샘플) ---")
print(f"Features: {llm_results[0]['features']}")
print(f"Reasoning: {llm_results[0]['reasoning']}")

# (JSON 파일 저장을 위해 import json이 필요하며, os는 파일 경로 확인에 사용됨)

# --- 5. 💾 LLM 추출 결과 데이터 저장 ---
OUTPUT_RESULTS_FILE = 'llm_costing_results.json'

print(f"\n5. 추출된 LLM 결과 데이터 저장 중... (파일: {OUTPUT_RESULTS_FILE})")

try:
    # JSON 파일로 저장
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
        # JSON 파일에 읽기 쉽도록 들여쓰기(indent=4)를 적용하여 저장합니다.
        json.dump(llm_results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ LLM 추출 결과 및 토큰 측정 데이터 저장 완료: {OUTPUT_RESULTS_FILE}")
    print(f"총 {len(llm_results)}개 레코드가 저장되었습니다.")
except Exception as e:
    print(f"⛔ JSON 파일 저장 중 오류 발생: {e}")