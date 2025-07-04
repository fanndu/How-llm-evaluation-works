import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.stats import norm, beta

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 加载预训练BERT模型和分词器
# 使用情感分析模型 "textattack/bert-base-uncased-SST-2"
# 这是一个在SST-2数据集上微调的BERT模型，用于二分类情感分析
# SST-2数据集标签: 0=负面, 1=正面
model_name = "textattack/roberta-base-ag-news"

# model_name = "bert-base-uncased"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
# Check if MPS (Metal Performance Shaders) is available on macOS
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)

print(f"Using device: {device}")

# Move model to appropriate device
model = model.to(device)

model.eval()

def randomized_smoothing(model, tokenizer, text, sigma=0.5, n_samples=5000, confidence=0.95):
    """
    改进的随机平滑，计算认证鲁棒性。
    参数：
        model: 预训练BERT模型
        tokenizer: BERT分词器
        text: 输入文本
        sigma: 高斯噪声标准差
        n_samples: 采样次数
        confidence: 置信度水平
    返回：
        predicted_class: 预测类别
        cert_radius: 认证半径
    """
    # 编码输入文本
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 获取词嵌入
    embedding_layer = model.roberta.embeddings
    with torch.no_grad():
        embeddings = embedding_layer(input_ids.to(device))

    # 存储采样预测结果
    predictions = []
    from tqdm import tqdm
    for _ in tqdm(range(n_samples), desc="Sampling", leave=False):
        noise = torch.normal(mean=0.0, std=sigma, size=embeddings.shape, device=embeddings.device)
        noisy_embeddings = embeddings + noise
        outputs = model(inputs_embeds=noisy_embeddings.to(device), attention_mask=attention_mask.to(device))
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred_class)

    # 统计预测结果
    predictions = np.array(predictions)
    class_counts = np.bincount(predictions, minlength=model.config.num_labels)
    most_common_class = np.argmax(class_counts)
    k_A = class_counts[most_common_class]
    
    # 使用Clopper-Pearson区间估计概率
    alpha = 1 - confidence
    p_A_lower = beta.ppf(alpha/2, k_A, n_samples - k_A + 1) if k_A < n_samples else 1.0
    k_B = np.max(class_counts[np.arange(len(class_counts)) != most_common_class]) if np.sum(class_counts) > k_A else 0
    p_B_upper = beta.ppf(1 - alpha/2, k_B + 1, n_samples - k_B) if k_B > 0 else 0.0

    # 计算认证半径
    if p_A_lower > p_B_upper:
        cert_radius = (sigma / 2) * (norm.ppf(p_A_lower) - norm.ppf(p_B_upper))
    else:
        cert_radius = 0.0

    return most_common_class, cert_radius

# 示例使用
text = "Reuters - As Shakespeare said, a rose by any other\name would smell as sweet. Right?"
sigma = 0.5  # 增加噪声强度
n_samples = 5000  # 增加采样次数

predicted_class, cert_radius = randomized_smoothing(model, tokenizer, text, sigma, n_samples)
print(f"预测类别: {predicted_class}")
print(f"认证半径: {cert_radius:.4f}")

# 参考文献
# [1] Zhang, Z., et al. (2023). Certified robustness for large language models with self-denoising. arXiv preprint arXiv:2307.07171.