from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# 1. 加载 SQuAD v2 验证集
dataset = load_dataset("squad", split="validation")

# 2. 加载 roberta-base 问答模型（这里用 HuggingFace自带的“deepset/roberta-base-squad2”，它是在SQuAD2上finetune的roberta-base）
model_ckpt = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 3. 加载SQuAD官方评测指标
squad_metric = evaluate.load("squad_v2")

# 4. 评测主循环（建议用小样本快速试跑，大批量建议用批处理优化）
predictions = []
references = []
for item in tqdm(dataset, desc="Evaluating"):
    question = item["question"]
    context = item["context"]
    pred = qa(question=question, context=context)
    predictions.append({
        "id": item["id"], 
        "prediction_text": pred["answer"], 
        "no_answer_probability": 0.0  # 或自定义打分
    })
    references.append({"id": item["id"], "answers": item["answers"]})

# 5. 计算分数
score = squad_metric.compute(predictions=predictions, references=references)
print("Roberta-base SQuAD2 验证集评测结果：")
print(score)