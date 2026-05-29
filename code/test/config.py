import os
import torch

# 프로젝트 최상위 경로 (code/test에서 상위 2단계)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === 데이터 경로 ===
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'subdataset', 'build_hetero_graph_v2.pt')
META_PATH = os.path.join(PROJECT_ROOT, 'output', 'knowledge_meta.json')
MASTER_FILE = os.path.join(PROJECT_ROOT, 'subdataset', 'arxiv_master_final.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'benchmark')

# === 장치 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 하이퍼파라미터 (모든 모델에 통일 적용하여 공정 비교 보장) ===
EMBEDDING_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.005
NUM_EPOCHS = 200
EVAL_INTERVAL = 5
SEED = 42
KNOWLEDGE_WEIGHT = 0.1 #식 임베딩 반영 강도 (과도한 노이즈 방지)

# === 평가 설정 ===
TOP_K = 20
EVAL_BATCH_SIZE = 4096

# === 데이터 분할 비율 ===
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (나머지 자동 할당)
