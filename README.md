📚 大语言模型评测配套项目（中文版）

本项目为《大语言模型评测》一书提供完整的配套代码、工具与数据集示例，帮助读者深入理解、动手实践，并掌握如何评测大语言模型的各项能力。

⸻

🚀 项目结构与内容

本书配套的代码仓库分为两个项目：

📌 项目一：交互式教学仓库（Jupyter Notebooks）

how-llm-evaluation-works
📗 对应书中各章节内容，逐步演示大语言模型的评测过程。

	•	✅ Jupyter Notebook形式，方便本地和Google Colab直接运行。
	•	✅ 每一章内容都独立成Notebook，提供详细代码注释和输出演示。
	•	✅ 覆盖鲁棒性、语义相似度、输出一致性、安全性、公平性等核心评测维度。

目录示例：

how-llm-evaluation-works/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_robustness_evaluation.ipynb
    ├── 02_semantic_similarity.ipynb
    ├── 03_output_consistency.ipynb
    └── 04_security_fairness.ipynb

📌 项目二：核心评测工具库（Python Package）

llm-eval-kit
🛠️ 独立的可复用评测模块，用于快速实现和扩展大语言模型评测任务。

	•	✅ 封装文本扰动、语义相似度计算、评测指标工具。
	•	✅ 支持主流大语言模型的评测（如GPT系列、LLaMA系列、Qwen等）。
	•	✅ 模块化设计，易于集成到其他评测或研究项目。

结构示例：

llm-eval-kit/
├── README.md
├── setup.py
├── llm_eval_kit/
│   ├── perturbation.py
│   ├── semantic_similarity.py
│   ├── evaluator.py
│   └── models.py
└── examples/
    └── quickstart.py


⸻

⚡ 快速上手指南

🖥️ 教学代码库快速启动：

# 克隆教学代码仓库
git clone https://github.com/fanndu/how-llm-evaluation-works
cd how-llm-evaluation-works
pip install -r requirements.txt

# 使用 Jupyter Lab 启动
jupyter lab

或使用 Google Colab 在线运行。

📦 工具库安装与使用：

# 克隆核心评测工具库
git clone https://github.com/fanndu/llm-eval-kit
cd llm-eval-kit

# 安装依赖
pip install -e .

# 运行示例
python examples/quickstart.py


⸻

🛠️ 依赖环境
	•	Python >= 3.10
	•	PyTorch
	•	HuggingFace Transformers
	•	sentence-transformers
	•	numpy, pandas

（详见每个仓库下的requirements.txt）

⸻

📜 许可证（License）

本书配套项目使用如下许可证：
	•	教学代码仓库：MIT License
	•	核心工具库：Apache License 2.0

⸻

