from openai import OpenAI
from datasets import load_from_disk
import json
import argparse
import base64
from tqdm import tqdm
import os
from PIL import Image, ImageEnhance
import re

client = OpenAI(
    base_url="http://47.242.151.133:24576/v1/",
    api_key="ml2025",
)

def load_data(path):
    return load_from_disk(path)

def preprocess_image(example):
    image = example["image"].convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.2)
    return image

def encode_image_to_base64(pil_img):
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, "tmp_img.png")
    pil_img.save(tmp_file, format="PNG")
    with open(tmp_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def normalize_prediction(text):
    text = text.strip()
    if text.isupper():
        words = text.split()
        text = " ".join(word.capitalize() for word in words)
    return text

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9$%.]+", " ", text)  # 保留基本字母和数字
    text = re.sub(r"\s+", " ", text)  # 多空格合一
    return text

def is_fuzzy_match(pred, golds):
    pred_norm = clean_text(pred)
    for gold in golds:
        gold_norm = clean_text(gold)
        # 精确或包含关系
        if pred_norm == gold_norm:
            return True
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return True
    return False

def generate_answer(example):
    image = preprocess_image(example)
    b64 = encode_image_to_base64(image)

    fewshot = [
        {"question": "What is the invoice number?", "answer": "INV-20240315"},
        {"question": "Who is the recipient of this document?", "answer": "John Doe"},
    ]

    system_prompt = (
        "You are a helpful document-reading assistant. "
        "Only return the answer to each question based on the document image. "
        "Do not include any explanations, punctuation, or extra words."
    )

    messages = [{"role": "system", "content": system_prompt}]

    for example_pair in fewshot:
        messages.append({"role": "user", "content": example_pair["question"]})
        messages.append({"role": "assistant", "content": example_pair["answer"]})

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": b64}},
            {"type": "text", "text": f"{example['question']}\nOnly return the answer. Do not add any other words."},
        ],
    })

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=messages,
    )

    return normalize_prediction(response.choices[0].message.content)

def evaluate_results(results):
    correct = 0
    for item in results:
        if is_fuzzy_match(item["prediction"], item["answers"]):
            correct += 1
    return round(correct / len(results), 2)

def main(args):
    ds = load_data(args.data_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = []

    for example in tqdm(ds, desc="Generating answers"):
        answer = generate_answer(example)
        results.append({"prediction": answer, "answers": example["answers"]})

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pass_rate = evaluate_results(results)
    print(f"Pass rate: {pass_rate}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="C:/Users/86159/Desktop/lab3/data-3/data/docvqa_100")
    parser.add_argument("--output_path", type=str, default="./results/docvqa_100_results_image_enhancement.json")
    args = parser.parse_args()
    main(args)
