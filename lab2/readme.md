# Lab2: Multi-hop QA System with LLM and RAG

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A multi-hop question answering system leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) technology.

## 📚 Background
This project explores the effectiveness of combining LLMs with RAG for answering multi-hop questions from the HotpotQA dataset. The system retrieves relevant context from documents and generates answers using hybrid retrieval methods (BM25 + semantic matching).

## 🎯 Task Objectives
1. Implement a RAG system for multi-hop QA
2. Evaluate with EM and F1 metrics
3. Analyze system performance
4. Propose and test improvement strategies

## 📊 Dataset
- **HotpotQA** (200 samples from LongBench)
- Fields: `id`, `question`, `context`, `answer`
- File: `hotpotqa_longbench.json`

## 🛠 Installation
```bash
git clone https://github.com/YuyaoFan/SE3323-MachineLearning/lab2.git
pip install -r requirements.txt
```

## 🚀 Usage
- Baseline System:
```bash
python main.py
```
- Improved System:
```bash
python main_improved.py
```

## 📈 Evaluation Metrics
**Exact Match (EM)**: Strict exact answer matching

**F1 Score**: Token-level overlap between prediction and ground truth

Expected baseline performance:
EM > 0.3 | F1 > 0.4

## 💡 Key Features
Hybrid retrieval (BM25 + semantic matching)

Context-aware chunking with sentence boundaries

Two implementations:

Baseline: Llama-3.1-8B-Instruct

Improved: Qwen-Plus with enhanced prompting

## 🛠️ Improvement Strategies
Enhanced prompt engineering

Better error handling for Yes/No questions

Context re-reading mechanism

Answer validation instructions

## ⚠️ Important Notes
- Replace API endpoints/keys in code for production use
- Results may vary based on model availability

