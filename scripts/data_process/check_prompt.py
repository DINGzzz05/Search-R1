import pandas as pd
import sys
import os

data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/nq_search"
train_path = os.path.join(data_dir, "train.parquet")

if not os.path.exists(train_path):
    print(f"❌ 文件不存在: {train_path}")
    sys.exit(1)

df = pd.read_parquet(train_path)
prompt = df['prompt'][0]

# 如果是列表（标准格式），取出 content
if isinstance(prompt, list) and len(prompt) > 0:
    content = prompt[0].get('content', '')
else:
    content = str(prompt)

print("=== 第一条 prompt 内容 ===")
print(content)