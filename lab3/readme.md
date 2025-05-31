# Lab3: Document Visual Question Answering with Vision-Language Models

## Author
范禹尧  
学号: 522070910015

## Overview
This project explores the performance of an open-source Vision-Language Model (Qwen2.5-VL-3B-Instruct) on document-based VQA tasks, specifically on DocVQA and MP-DocVQA datasets. Various optimization strategies were tested to improve model accuracy.
## Dependencies
- Python 3.8+
- `datasets` library
- Qwen2.5-VL API (provided by course server)
- Other packages as listed in individual `.py` files

## DocVQA Experiments

### 1. Baseline
- Directly feeds RGB image and question to VLM
- Simple prompt format
- **Pass Rate**: 88%

### 2. Prompt Engineering
- Few-shot example + explicit instruction prompt
- Clarifies response format
- **Pass Rate**: 87%

### 3. Image Enhancement
- Enhances contrast while preserving key structures
- Introduces normalization and fuzzy matching postprocessing
- **Pass Rate**: 94%

## MP-DocVQA Experiments

### 1. Baseline (image1 only)
- Uses only the first image of each sample
- **Pass Rate**: 41%

### 2. Answer Page Index
- Uses `answer_page_idx` to locate the relevant image
- **Pass Rate**: 44%

### 3. Multi-Image Concatenation + Prompt Engineering
- Selects up to 4 pages centered around answer page
- Combines them into a 2x2 layout (max 840x840)
- Enhances prompt to clarify presence of answer
- **Pass Rate**: 58%

## Notes
- All results are based on 100 sampled examples from each dataset.
- All `.json` result files contain raw model predictions for evaluation.
- No data or models are included in the submission.

## Reference
- [DocVQA dataset](https://huggingface.co/datasets/lmms-lab/DocVQA)
- [MP-DocVQA dataset](https://huggingface.co/datasets/lmms-lab/MP-DocVQA)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
