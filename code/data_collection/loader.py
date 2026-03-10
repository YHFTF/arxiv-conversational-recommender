import torch
_real_torch_load = torch.load

def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 기본값을 False로 되돌림
    return _real_torch_load(*args, **kwargs)

torch.load = _torch_load_with_weights_only_false  # 임시 패치(세션 동안만)
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name="ogbn-arxiv")

print("데이터셋이 성공적으로 로드되었습니다.")
print("스크립트 종료")