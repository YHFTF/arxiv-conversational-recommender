import os
import sys
import subprocess


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    keyword_script = os.path.join(current_dir, "llm_keyword_extraction.py")
    label_pred_script = os.path.join(current_dir, "llm_pred_label.py")

    if not os.path.exists(keyword_script):
        print(f"[Error] 키워드 추출 스크립트를 찾을 수 없습니다: {keyword_script}")
        return
    if not os.path.exists(label_pred_script):
        print(f"[Error] 라벨 예측 스크립트를 찾을 수 없습니다: {label_pred_script}")
        return

    print("=" * 60)
    print("  LLM 비용 측정 실행 프로그램")
    print("=" * 60)
    print("  1. 키워드 추출 비용 측정 (llm_keyword_extraction 프롬프트 기준)")
    print("  2. 라벨 예측 비용 측정 (llm_pred_label 프롬프트 기준)")
    print("  q. 종료")
    print("=" * 60)

    while True:
        choice = input(">> 모드를 선택하세요 (1/2/q): ").strip().lower()

        if choice == "1":
            print("\n[Mode 1] 키워드 추출 비용 측정을 시작합니다...\n")
            env = os.environ.copy()
            env["LLM_COSTING_MODE"] = "keyword"
            # 현재 파이썬 실행 파일로 llm_keyword_extraction.py 실행 (코스트 모드)
            subprocess.run([sys.executable, keyword_script], check=False, env=env)
            break

        elif choice == "2":
            print("\n[Mode 2] 라벨 예측 비용 측정을 시작합니다...\n")
            env = os.environ.copy()
            env["LLM_COSTING_MODE"] = "label"
            # 현재 파이썬 실행 파일로 llm_pred_label.py 실행 (코스트 모드)
            subprocess.run([sys.executable, label_pred_script], check=False, env=env)
            break

        elif choice == "q":
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 입력입니다. 1, 2, q 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()



