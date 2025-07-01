# RoBERTa-base SQuAD v2 微调

本目录包含在 SQuAD v2 数据集上对 RoBERTa-base 模型进行微调的完整代码。

## 文件说明

- `fine_tune_roberta_squad2.py` - 主要的微调脚本
- `README.md` - 本说明文档

## 环境依赖

在运行脚本前，请确保安装以下依赖：

```bash
pip install transformers datasets evaluate torch accelerate
```

如果有GPU，建议安装CUDA版本的PyTorch以加速训练。

## 使用方法

### 基本使用

```bash
cd notebooks/robustness/training
python fine_tune_roberta_squad2.py
```

### 快速测试

如果只是想测试代码是否正常运行，可以修改脚本中的参数：

```python
# 在 fine_tune_roberta_squad2.py 中修改
USE_SMALL_DATASET = True  # 使用小数据集
SMALL_DATASET_SIZE = 1000  # 只使用1000个样本
```

## 配置参数

可以在脚本开头的配置部分调整以下参数：

```python
# 模型配置
MODEL_NAME = "roberta-base"  # 可改为其他模型
OUTPUT_DIR = "./roberta-squad2-finetuned"  # 输出目录

# 训练配置
TRAIN_BATCH_SIZE = 8      # 训练批量大小（根据显存调整）
EVAL_BATCH_SIZE = 8       # 评估批量大小
LEARNING_RATE = 3e-5      # 学习率
NUM_EPOCHS = 2            # 训练轮数
MAX_LENGTH = 384          # 最大序列长度
```

## 训练时间预估

- **CPU**: 使用完整数据集可能需要数小时到数天
- **GPU**: 
  - 单卡GPU (如 RTX 3080): 约2-4小时
  - 高端GPU (如 A100): 约1-2小时

## 输出文件

训练完成后，以下文件将保存在输出目录中：

```
roberta-squad2-finetuned/
├── config.json          # 模型配置
├── pytorch_model.bin     # 模型权重
├── tokenizer_config.json # 分词器配置
├── tokenizer.json        # 分词器文件
├── vocab.json           # 词汇表
├── merges.txt           # BPE合并规则
├── eval_results.txt     # 评估结果
└── logs/                # 训练日志
```

## 使用微调后的模型

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 加载微调后的模型
tokenizer = AutoTokenizer.from_pretrained("./roberta-squad2-finetuned")
model = AutoModelForQuestionAnswering.from_pretrained("./roberta-squad2-finetuned")

# 进行推理
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# 解析答案
start_logits = outputs.start_logits
end_logits = outputs.end_logits

start_idx = start_logits.argmax()
end_idx = end_logits.argmax()

answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1])
print(f"答案: {answer}")
```

## 注意事项

1. **内存要求**: 确保有足够的内存加载数据集和模型
2. **GPU显存**: 如果显存不足，可以减小 `TRAIN_BATCH_SIZE`
3. **数据下载**: 首次运行会下载SQuAD v2数据集，需要网络连接
4. **训练时间**: 完整训练需要较长时间，建议先用小数据集测试

## 故障排除

### 常见错误及解决方案

1. **CUDA out of memory**
   - 减小 `TRAIN_BATCH_SIZE` 和 `EVAL_BATCH_SIZE`
   - 设置 `USE_SMALL_DATASET = True`

2. **网络连接问题**
   - 确保能访问 Hugging Face Hub
   - 考虑使用镜像或离线模式

3. **依赖包问题**
   - 更新到最新版本: `pip install --upgrade transformers datasets`

## 性能优化建议

1. **使用混合精度训练**: 脚本已自动启用（GPU环境）
2. **调整批量大小**: 在显存允许的情况下尽量增大
3. **使用多GPU**: 可以通过 `accelerate` 库实现分布式训练
4. **梯度累积**: 如果批量大小受限，可以使用梯度累积

## 参考资料

- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [SQuAD v2 数据集](https://rajpurkar.github.io/SQuAD-explorer/)
- [RoBERTa 论文](https://arxiv.org/abs/1907.11692) 