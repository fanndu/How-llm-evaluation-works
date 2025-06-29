from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# 1. 加载 BERT 模型和分词器（以情感分析为例）
model_name = "roberta-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# 2. 定义模型预测函数：输入 list[str]，输出 numpy array of shape (batch, num_classes)
def predict_fn(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

# 3. 定义 TextFooler 风格的词重要性评分函数
def compute_word_importance_scores(text, tokenizer, model_predict_fn, unk_token="[UNK]"):
    tokens = tokenizer.tokenize(text)
    len_text = len(tokens)

    orig_probs = model_predict_fn([text])[0]
    orig_label = np.argmax(orig_probs)
    orig_conf = orig_probs[orig_label]

    leave1_texts = []
    for i in range(len_text):
        tokens_copy = tokens[:i] + [unk_token] + tokens[i+1:]
        leave1_texts.append(tokenizer.convert_tokens_to_string(tokens_copy))

    leave1_probs = model_predict_fn(leave1_texts)
    leave1_preds = np.argmax(leave1_probs, axis=1)
    leave1_confs = leave1_probs[np.arange(len_text), orig_label]
    max_other_confs = np.max(leave1_probs, axis=1)

    importance_scores = (
        orig_conf - leave1_confs
        + (leave1_preds != orig_label) * (max_other_confs - leave1_confs)
    )

    return importance_scores.tolist(), tokens

# 4. 测试用例
text = "This was the worst film I've ever seen in my life."
scores, tokens = compute_word_importance_scores(text, tokenizer, predict_fn)

# 5. 输出结果
print("Token Importance:")
for token, score in zip(tokens, scores):
    print(f"{token:15s} -> {score:.4f}")