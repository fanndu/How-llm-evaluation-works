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

## 评估结果
原始数据：{'exact': 100.0, 'f1': 100.0, 'total': 4827, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0, 'HasAns_total': 4827, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0, 'best_f1_thresh': 0.0}

扰动后的数据：{'exact': 86.22332711829293, 'f1': 89.88948599432614, 'total': 4827, 'HasAns_exact': 86.22332711829293, 'HasAns_f1': 89.88948599432614, 'HasAns_total': 4827, 'best_exact': 86.22332711829293, 'best_exact_thresh': 0.0, 'best_f1': 89.88948599432614, 'best_f1_thresh': 0.0}



## 数据说明

本实验使用的数据集规模:

- squad2-filter/filtered-squad.parquet: 从 SQuAD v2 validation 集的前 100 条数据中筛选出模型能正确回答的样本
- perturbed/squad2-question-perturbed.parquet: 基于 filtered-squad.parquet 生成的扰动数据集,保持答案不变但对问题进行了同义词替换等扰动

这两个数据集规模较小,主要用于验证模型对问题扰动的鲁棒性。如需进行更大规模的评估,可以扩大原始数据的采样范围。



# TODO
## 进度显示

为了更好地监控评估进度,各脚本都添加了进度条显示:

- 使用 tqdm 显示数据处理和评估的进度
- 显示已处理样本数/总样本数
- 预估剩余时间

## 参数配置

主要参数都通过命令行参数传入:





 