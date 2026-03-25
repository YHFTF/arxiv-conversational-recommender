import json
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

OA_FILE = os.path.join(project_root, 'output', 'author_data_openalex.json')
LLM_FILE = os.path.join(project_root, 'output', '3llm_extraction_results.json')
TSV_FILE = os.path.join(project_root, 'subdataset', 'titleabs.tsv')
OUTPUT_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')

def build_final_json_brute_force():
    print("🚀 [데이터 통합 시작: OpenAlex + TSV + LLM]")

    # 1. TSV 로드 (ID를 키로, 제목과 초록을 저장)
    id_to_content = {}
    print(f"📖 TSV 파일을 인덱싱 중... ({TSV_FILE})")
    
    try:
        with open(TSV_FILE, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                # 탭(\t)으로 분리: [0]=ID, [1]=Title, [2]=Abstract
                parts = line.split('\t')
                
                if len(parts) >= 2:
                    p_id = parts[0].strip()
                    title = parts[1].strip()
                    # 초록이 없을 경우를 대비해 예외 처리
                    abstract = parts[2].strip() if len(parts) > 2 else ""
                    
                    id_to_content[p_id] = {
                        "title": title,
                        "abstract": abstract
                    }
    except FileNotFoundError:
        print(f"❌ 에러: {TSV_FILE} 파일을 찾을 수 없습니다.")
        return

    # 2. JSON 데이터 로드
    with open(OA_FILE, 'r', encoding='utf-8') as f:
        oa_raw = json.load(f)
    
    # LLM 파일이 없을 경우를 대비해 빈 리스트로 처리
    try:
        with open(LLM_FILE, 'r', encoding='utf-8') as f:
            llm_raw = json.load(f)
    except FileNotFoundError:
        print("⚠️ 경고: LLM 추출 결과 파일이 없습니다. 기본값으로 채웁니다.")
        llm_raw = []
    
    llm_map = {item['node_idx']: item for item in llm_raw}

    # 3. 통합 작업
    final_master_list = []
    success_count = 0

    print(f"🔗 ID 매칭 중 (대상 데이터: {len(oa_raw)}건)...")
    for oa_item in oa_raw:
        n_idx = oa_item['node_idx']
        # paper_id를 문자열로 변환하여 TSV 키와 맞춤
        p_id = str(oa_item['paper_id']).strip()
        
        # TSV에서 데이터 검색
        content = id_to_content.get(p_id)
        
        if content:
            title = content['title']
            abstract = content['abstract']
            success_count += 1
        else:
            title = "Unknown"
            abstract = "Unknown"

        # LLM 지식 정보 매칭
        knowledge = llm_map.get(n_idx, {})

        final_master_list.append({
            "node_idx": n_idx,
            "paper_id": p_id, # 원래 필드명 유지 (또는 arxiv_id)
            "title": title,
            "abstract": abstract,
            "authors": oa_item.get('authors', []),
            "knowledge": {
                "domain": knowledge.get("domain", []),
                "task": knowledge.get("task", []),
                "method": knowledge.get("method", [])
            }
        })

    # 4. 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_master_list, f, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"✅ 통합 완료!")
    print(f" - 매칭 성공(ID 일치): {success_count} / {len(oa_raw)}")
    print(f" - 결과 파일: {OUTPUT_FILE}")
    print("="*50)

    # 샘플 출력
    if final_master_list:
        sample = final_master_list[0]
        for item in final_master_list:
            if item['title'] != "Unknown":
                sample = item
                break
        print(f"\n🧐 [추출 샘플 확인 - ID: {sample['paper_id']}]")
        print(f"제목: {sample['title']}")
        print(f"초록 요약: {sample['abstract'][:100]}...")

if __name__ == "__main__":
    build_final_json_brute_force()