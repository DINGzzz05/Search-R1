'''
改写提示词
'''


import pandas as pd
import os
import sys

NEW_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "IMPORTANT: When you write the final answer inside <answer> tags, you MUST cite the sources "
    "using the exact reference numbers shown in the search results (e.g., [1], [2]). "
    "Place each citation directly after the information it supports. "
    'For instance, if Search Result [1] says "Paris is the capital of France", '
    'your answer should contain "Paris is the capital of France[1]". '
    "Question: {question}\n"
)

def extract_question(prompt_list):
    """稳健地提取问题文本"""
    # prompt_list 是类似 [{'role': 'user', 'content': '...'}] 的列表
    if isinstance(prompt_list, list) and len(prompt_list) > 0:
        for msg in prompt_list:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                marker = "Question: "
                idx = content.rfind(marker)
                if idx != -1:
                    question = content[idx + len(marker):].strip()
                    return question
    # 如果无法提取，返回空字符串（至少不会让训练崩溃）
    return ""


def rewrite_prompt(prompt_list):
    question = extract_question(prompt_list)
    # 调试输出（只打印前几个字符）
    if len(question) < 5:
        print(f"⚠️ 警告：提取到的问题过短：'{question}'，原始prompt：{prompt_list}")
    new_content = NEW_TEMPLATE.format(question=question)
    return [{"role": "user", "content": new_content}]


def process_file(filepath, backup=True):
    if backup and os.path.exists(filepath):
        backup_path = filepath.replace(".parquet", "_backup.parquet")
        if not os.path.exists(backup_path):
            df = pd.read_parquet(filepath)
            df.to_parquet(backup_path, index=False)
            print(f"📦 已备份: {backup_path}")

    print(f"✍️  重写中: {filepath}")
    df = pd.read_parquet(filepath)
    df["prompt"] = df["prompt"].apply(rewrite_prompt)
    df.to_parquet(filepath, index=False)
    print(f"✅ 完成: {filepath}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/nq_search"
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    for path in [train_path, test_path]:
        if os.path.exists(path):
            process_file(path, backup=True)
        else:
            print(f"⚠️ 文件不存在: {path}")
    print("\n🎉 全部完成。")