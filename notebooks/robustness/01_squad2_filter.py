from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import os

# 1. 只加载有答案的样本
dataset = load_dataset("squad_v2", split="validation")

has_ans_dataset = [item for item in dataset if len(item['answers']['text']) > 0]

# 2. 加载问答模型
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# 3. 评估，只保留模型能答对（EM=1）的样本
def is_exact_match(pred, golds):
    pred = pred.strip()
    return any(pred == gold.strip() for gold in golds)

filtered_data = []
for item in has_ans_dataset:
    context = item["context"]
    question = item["question"]
    gold_answers = item["answers"]["text"]

    result = qa(question=question, context=context)
    pred_text = result["answer"]

    if is_exact_match(pred_text, gold_answers):
        filtered_data.append(item)  # 保留原始格式

print(f"总共保留 {len(filtered_data)} 条模型能答对的有答案数据。")

# 4. 整理为 HuggingFace SQuAD v2 parquet 兼容格式
rows = []
for item in filtered_data:
    row = {
        "id": item["id"],
        "title": item.get("title", ""),  # 保留title字段，如无可留空
        "context": item["context"],
        "question": item["question"],
        "answers": {
            "text": item["answers"]["text"],
            "answer_start": item["answers"]["answer_start"]
        },
        "is_impossible": False
    }
    rows.append(row)

df = pd.DataFrame(rows)

# 5. 保存为 parquet
save_dir = "squad2-filter"  # 你可以修改为目标目录
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "filtered-squad.parquet")
df.to_parquet(save_path, engine="pyarrow", index=False)
print(f"已保存为 Parquet 格式：{save_path}")

# 检查输出
df_check = pd.read_parquet(save_path)
print(df_check.head())