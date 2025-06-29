import jieba
import numpy as np

def compute_word_importance_scores_chinese(text, model_predict_fn):
    """
    适配中文 + jieba 分词的 TextFooler 风格词组级别重要性计算

    参数：
        text: 原始中文字符串
        model_predict_fn: 预测函数，输入文本列表 -> 返回 logits 或概率矩阵 (batch, num_classes)

    返回：
        importance_scores: List[float]，每个词组的重要性
        tokens: List[str]，分词后的词组序列
    """
    print("compute_word_importance_scores_chinese text", text)
    # 中文词组级分词
    tokens = list(jieba.cut(text))
    len_text = len(tokens)

    # 原始模型预测
    orig_probs = model_predict_fn([text])[0]
    orig_label = np.argmax(orig_probs)
    orig_conf = orig_probs[orig_label]

    # Leave-one-out：逐个替换词为"[UNK]"（或无效词）
    leave1_texts = []
    for i in range(len_text):
        tokens_copy = tokens[:i] + ["[UNK]"] + tokens[i+1:]
        perturbed_text = "".join(tokens_copy)
        leave1_texts.append(perturbed_text)

    # 批量预测 perturbed 结果
    leave1_probs = model_predict_fn(leave1_texts)
    leave1_preds = np.argmax(leave1_probs, axis=1)
    leave1_confs = leave1_probs[np.arange(len_text), orig_label]
    max_other_confs = np.max(leave1_probs, axis=1)

    # Importance score: 原置信度下降 + 类别跳变惩罚项
    importance_scores = (
        orig_conf - leave1_confs
        + (leave1_preds != orig_label) * (max_other_confs - leave1_confs)
    )

    return importance_scores.tolist(), tokens

from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# 中文预测函数


def predict_fn(text_list):
    print("predict_fn text_list", text_list)
    """预测函数，用于批量预测文本的情感分类概率
    
    Args:
        text_list (List[str]): 输入的中文文本列表
        
    Returns:
        numpy.ndarray: 形状为(batch_size, num_classes)的概率矩阵,
                      每行表示一个样本在各个类别上的预测概率
    """
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

text = "这部影片是我人生中看过的最糟糕的电影。"
scores, tokens = compute_word_importance_scores_chinese(text, predict_fn)

print("词组级 Token Importance:")
for token, score in zip(tokens, scores):
    print(f"{token:<10} -> {score:.4f}")