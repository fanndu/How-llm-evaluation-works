from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator
)
from datasets import load_dataset
import torch
import evaluate
import numpy as np

# 1. 设备配置（自动使用 MPS / CUDA / CPU）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# 2. 超参数
model_checkpoint = "roberta-base"
batch_size = 8
n_epochs = 2
max_length = 384
doc_stride = 128
learning_rate = 3e-5

# 3. 加载数据
raw_datasets = load_dataset("squad_v2")

# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from transformers import Trainer
import torch.nn as nn

class QA_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]

        # 损失函数（交叉熵）
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        return (loss, outputs) if return_outputs else loss

# 5. Tokenize 函数
def preprocess_function(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

# 6. 模型
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model.to(device)

# 7. Training 参数
args = TrainingArguments(
    "qa-roberta-squadv2",
    eval_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=n_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
)

# 8. Metric 评估函数（使用 HuggingFace evaluate 的 squad_v2）
squad_metric = evaluate.load("squad_v2")

def compute_metrics(p):
    return squad_metric.compute(predictions=p.predictions, references=p.label_ids)

# 9. Trainer
trainer = QA_Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 10. 训练与评估
trainer.train()
eval_results = trainer.evaluate()

print("Evaluation Results:")
print(eval_results)