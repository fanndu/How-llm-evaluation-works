# 模型鲁棒性评估

本目录包含了对问答模型鲁棒性的评估实验。主要分为以下几个步骤:

## 1. 数据过滤 (01_filter_data.py)

对SQuAD v2数据集进行过滤,只保留模型能够正确回答的样本:

- 使用deepset/roberta-base-squad2模型进行评估
- 计算每个问题的F1和Exact Match分数
- 只保留F1分数>0.8的样本作为高质量数据

## 2. 数据扰动 (02_perturb_data.py) 

对过滤后的高质量数据进行扰动:

- 同义词替换
- 拼写错误注入
- 问题改写
- 保持答案不变

## 3. 鲁棒性评估 (03_evaluate.py)

对比原始问题和扰动问题的模型表现:

- 计算F1、Exact Match等指标
- 分析模型对不同类型扰动的敏感度
- 评估模型鲁棒性

## 运行方式

按顺序执行以下脚本:
- 01_squad2_filter.py
- 02_perpurbation.py
- 03_squad_eval.py







 