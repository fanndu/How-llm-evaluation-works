from datasets import load_dataset
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.augmentation import Augmenter
import pandas as pd
import os

# 1. 加载 SQuAD v2 validation 数据集
dataset = load_dataset("parquet", data_files="squad2-filter/filtered-squad.parquet", split="train")

# 2. 初始化 TextFooler 变换
transformation = WordSwapEmbedding(max_candidates=50)
constraints = [RepeatModification(), StopwordModification()]
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.2,
    transformations_per_example=1
)

# 3. 扰动所有 question 字段
perturbed_questions = []
for item in dataset:
    orig_q = item["question"]
    perturbed = augmenter.augment(orig_q)
    perturbed_questions.append(perturbed[0] if perturbed else orig_q)

# 4. 构建新的 DataFrame，字段结构与 SQuAD v2 保持一致
rows = []
for i, item in enumerate(dataset):
    row = {
        "id": item["id"],
        "title": item.get("title", ""),
        "context": item["context"],
        "question": perturbed_questions[i],  # 替换为扰动后问题
        "answers": {
            "text": item["answers"]["text"],
            "answer_start": item["answers"]["answer_start"]
        },
        "is_impossible": item.get("is_impossible", False)
    }
    rows.append(row)

df = pd.DataFrame(rows)

# 5. 保存为 Parquet
save_dir = "perturbed"  # 指定你的目录
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "squad2-question-perturbed.parquet")
df.to_parquet(save_path, engine="pyarrow", index=False)
print(f"扰动后数据已保存为 Parquet: {save_path}")