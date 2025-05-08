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
git clone https://github.com/YuyaoFan/multi-hop-qa-rag.git
cd multi-hop-qa-rag
pip install -r requirements.txt
