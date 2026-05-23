'''
为提示词添加引用要求，确保模型在回答时正确引用搜索结果中的信息
注意备份需要空间
'''

import pandas as pd
import sys, os

MARKER = "For example, <answer> Beijing </answer>."
INSERT = (
    " IMPORTANT: When you write the final answer inside <answer> tags, "
    "you MUST cite the sources using the exact reference numbers shown in the search results (e.g., [1], [2]). "
    "Place each citation directly after the information it supports. "
    'For instance, if Search Result [1] says "Paris is the capital of France", '
    'your answer should contain "Paris is the capital of France[1]".'
)

def process_file(path):
    df = pd.read_parquet(path)
    new_prompts = []
    for p in df['prompt']:
        content = p[0]['content']
        if MARKER in content:
            new_content = content.replace(MARKER, MARKER + INSERT, 1)
        else:
            new_content = content.strip() + "\n" + INSERT.strip()
        new_prompts.append([{"role": "user", "content": new_content}])
    df['prompt'] = new_prompts
    df.to_parquet(path, index=False)
    print(f"✅ {path} updated.")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/nq_search"
    for split in ["train", "test"]:
        path = os.path.join(data_dir, f"{split}.parquet")
        if os.path.exists(path):
            process_file(path)