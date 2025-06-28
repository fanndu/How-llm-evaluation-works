from datasets import load_dataset
from transformers import pipeline
import evaluate

# dataset = load_dataset("parquet", data_files="squad2-filter/filtered-squad.parquet", split="train")
dataset = load_dataset("parquet", data_files="perturbed/squad2-question-perturbed.parquet", split="train")
#distilbert/distilbert-base-uncased-distilled-squad for comparison
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

predictions = []
references = []

for item in dataset:
    context = item["context"]
    question = item["question"]
    id_ = item["id"]
    result = qa_model({"context": context, "question": question})
    pred_text = result['answer']
    # 规范化输出格式
    predictions.append({
        "id": id_,
        "prediction_text": pred_text,
        "no_answer_probability": 0.0  # 如无概率信息，默认填0.0
    })
    references.append({
        "id": id_,
        "answers": item["answers"]    # 直接用原始答案字典
    })

squad_metric = evaluate.load("squad_v2")
results = squad_metric.compute(predictions=predictions, references=references)
print(results)