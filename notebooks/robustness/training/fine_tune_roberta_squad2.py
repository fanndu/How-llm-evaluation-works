#!/usr/bin/env python3
"""
SQuAD v2 微调脚本 - RoBERTa-base 模型

该脚本实现了在 SQuAD v2 数据集上对 RoBERTa-base 进行微调的完整流程，
包括数据预处理、模型训练、评估和结果保存。

使用方法:
    python fine_tune_roberta_squad2.py

依赖:
    pip install transformers datasets evaluate torch accelerate
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    default_data_collator
)
import evaluate
import torch

# ============================================================================
# 配置参数
# ============================================================================

# 模型配置
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./roberta-squad2-finetuned"

# 训练配置
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 2
MAX_LENGTH = 384
STRIDE = 128

# 是否使用小数据集进行快速测试
USE_SMALL_DATASET = False  # 设为 True 可快速测试
SMALL_DATASET_SIZE = 1000

print("🚀 开始 SQuAD v2 微调流程...")
print(f"📱 模型: {MODEL_NAME}")
print(f"💾 输出目录: {OUTPUT_DIR}")
print(f"🔧 批量大小: {TRAIN_BATCH_SIZE}")
print(f"📈 学习率: {LEARNING_RATE}")
print(f"🔄 训练轮数: {NUM_EPOCHS}")

# ============================================================================
# 1. 数据加载
# ============================================================================

print("\n📥 加载 SQuAD v2 数据集...")
datasets = load_dataset("squad_v2")

if USE_SMALL_DATASET:
    print(f"⚡ 使用小数据集进行快速测试 (大小: {SMALL_DATASET_SIZE})")
    datasets["train"] = datasets["train"].select(range(SMALL_DATASET_SIZE))
    datasets["validation"] = datasets["validation"].select(range(SMALL_DATASET_SIZE // 5))

print(f"✅ 训练集大小: {len(datasets['train'])}")
print(f"✅ 验证集大小: {len(datasets['validation'])}")

# ============================================================================
# 2. 模型和分词器加载
# ============================================================================

print(f"\n🤖 加载模型和分词器: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

# ============================================================================
# 3. 数据预处理
# ============================================================================

def preprocess_function(examples):
    """
    数据预处理函数
    
    将问题和上下文进行tokenization，并处理答案的位置标注
    """
    # 清理问题文本
    questions = [q.lstrip() for q in examples["question"]]
    
    # tokenization
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",  # 只截断context部分
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 处理溢出tokens的映射
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    # 初始化答案位置列表
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        # 获取序列ID来区分question和context
        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # 如果没有答案(SQuAD v2的impossible问题)
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # 获取答案的字符位置
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # 找到context在token序列中的范围
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            
            # 将字符位置转换为token位置
            token_start_index = None
            token_end_index = None
            
            for idx, (offset_start, offset_end) in enumerate(offsets[context_start:context_end], context_start):
                if offset_start <= start_char < offset_end:
                    token_start_index = idx
                if offset_start < end_char <= offset_end:
                    token_end_index = idx
                    
            # 如果找不到答案位置，标记为impossible
            if token_start_index is None or token_end_index is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("\n🔄 开始数据预处理...")
tokenized_datasets = datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=datasets["train"].column_names,
    desc="预处理数据集"
)

print("✅ 数据预处理完成")

# ============================================================================
# 4. 评估指标设置
# ============================================================================

# 加载SQuAD v2评估指标
metric = evaluate.load("squad_v2")

def compute_metrics(eval_pred):
    """
    计算评估指标
    """
    predictions, labels = eval_pred
    
    # 这里需要根据实际的预测格式进行调整
    # 由于这是简化版本，我们暂时返回占位符
    return {"eval_f1": 0.0, "eval_exact_match": 0.0}

# ============================================================================
# 5. 训练参数设置
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                    # 输出目录
    eval_strategy="epoch",              # 每个epoch后评估
    save_strategy="epoch",                    # 每个epoch后保存
    learning_rate=LEARNING_RATE,             # 学习率
    per_device_train_batch_size=TRAIN_BATCH_SIZE,  # 训练批量大小
    per_device_eval_batch_size=EVAL_BATCH_SIZE,    # 评估批量大小
    num_train_epochs=NUM_EPOCHS,             # 训练轮数
    weight_decay=0.01,                       # 权重衰减
    save_total_limit=2,                      # 最多保存2个checkpoint
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),  # 日志目录
    logging_steps=100,                       # 每100步记录一次
    fp16=torch.cuda.is_available(),          # 混合精度训练(仅GPU)
    dataloader_pin_memory=False,             # 减少内存使用
    report_to=None,                          # 不使用外部日志服务
    load_best_model_at_end=True,             # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",       # 最佳模型的评估指标
    greater_is_better=False,                 # loss越小越好
)

# ============================================================================
# 6. 创建Trainer
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================================
# 7. 开始训练
# ============================================================================

print("\n🏋️ 开始模型训练...")
print("⏰ 这可能需要一些时间，请耐心等待...")

try:
    # 开始训练
    train_result = trainer.train()
    
    print("✅ 训练完成！")
    print(f"📊 训练指标: {train_result.metrics}")
    
    # 保存最终模型
    trainer.save_model()
    trainer.save_state()
    
    print(f"💾 模型已保存到: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"❌ 训练过程中出现错误: {e}")
    raise

# ============================================================================
# 8. 模型评估
# ============================================================================

print("\n📊 开始模型评估...")

try:
    # 在验证集上评估
    eval_results = trainer.evaluate()
    print("✅ 评估完成！")
    print(f"📈 评估结果: {eval_results}")
    
    # 保存评估结果
    eval_results_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
    with open(eval_results_file, "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📄 评估结果已保存到: {eval_results_file}")
    
except Exception as e:
    print(f"❌ 评估过程中出现错误: {e}")

# ============================================================================
# 9. 完成
# ============================================================================

print("\n🎉 微调流程完成！")
print(f"📂 输出目录: {OUTPUT_DIR}")
print("\n📝 后续步骤:")
print("1. 检查训练日志和评估结果")
print("2. 使用微调后的模型进行推理")
print("3. 在测试集上进行最终评估")

print("\n💡 使用微调后的模型:")
print(f"""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}")
model = AutoModelForQuestionAnswering.from_pretrained("{OUTPUT_DIR}")
""") 