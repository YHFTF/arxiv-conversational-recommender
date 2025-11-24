import pandas as pd
import torch

mapping_path = "dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz"
df = pd.read_csv(mapping_path, compression="gzip")

print("총 레이블 개수:", len(df))       # 40일 것
print(df)