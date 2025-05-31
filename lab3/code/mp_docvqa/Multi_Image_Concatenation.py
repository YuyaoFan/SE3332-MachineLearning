import os
import json
import base64
import argparse
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from datasets import load_from_disk
from openai import OpenAI

REQUEST_TIMEOUT = 30

client = OpenAI(
    base_url="base_url",
    api_key="api_key",
)

def load_data(path):
    return load_from_disk(path)

def resize_image(image, target_size=(420, 420)):
    return image.resize(target_size, Image.BICUBIC)

def select_pages(example, window=2):
    """选取答案页及其前后页（最多4页）"""
    try:
        idx = int(example.get("answer_page_idx", 1))
    except:
        idx = 1

    indices = list(range(max(1, idx - window), min(20, idx + window + 1)))
    selected = []
    for i in indices:
        key = f"image_{i}"
        if key in example and example[key] is not None:
            selected.append(example[key])
        if len(selected) == 4:
            break
    return selected

def preprocess_image(example):
    images = select_pages(example)
    if not images:
        return None
    images = [resize_image(img).convert("RGB") for img in images]
    while len(images) < 4:
        images.append(Image.new('RGB', (420, 420), (255, 255, 255)))
    grid = Image.new('RGB', (840, 840), (255, 255, 255))
    for i, img in enumerate(images):
        x = (i % 2) * 420
        y = (i // 2) * 420
        grid.paste(img, (x, y))
    return grid

def generate_answer(example):
    grid_image = preprocess_image(example)
    if grid_image is None:
        return "ERROR: No valid images"

    tmp_file = "./tmp/grid.png"
    os.makedirs("./tmp", exist_ok=True)
    grid_image.save(tmp_file, format="PNG")

    with open(tmp_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    base64_image = f"data:image;base64,{encoded}"

    prompt = (
        f"Question: {example['question']}\n"
        "The image below shows the most relevant pages from a multi-page document, including the page that contains the answer.\n"
        "Please read all information carefully and return only the correct answer. If the answer is not present, say 'UNKNOWN'."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-3B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a multi-page document question answering assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_image}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=60,
                temperature=0.1,
                timeout=REQUEST_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error (attempt {attempt+1}): {e}")
            import time
            time.sleep(2 ** attempt)
    return "ERROR: API request failed"

def evaluate_results(results):
    score = 0
    for result in results:
        if result["generation"].lower() in result["answers"].lower():
            score += 1
    return round(score / len(results), 2)

def main(args):
    if not args.eval_only:
        ds = load_data(args.data_path)
        results = []
        start_idx = 0
        if os.path.exists(args.output_path):
            with open(args.output_path, "r") as f:
                results = json.load(f)
                start_idx = len(results)

        for i in tqdm(range(start_idx, len(ds)), desc="Generating answers"):
            example = ds[i]
            answer = generate_answer(example)
            results.append({
                "generation": answer,
                "answers": example['answers'],
                "question": example['question'],
                "index": i
            })
            if (i+1) % 5 == 0:
                with open(args.output_path, "w") as f:
                    json.dump(results, f)
        with open(args.output_path, "w") as f:
            json.dump(results, f)
        print(f"Pass rate: {evaluate_results(results)}")
    else:
        with open(args.output_path, "r") as f:
            results = json.load(f)
        print(f"Pass rate: {evaluate_results(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="C:/Users/86159/Desktop/lab3/data-3/data/mp_docvqa_100")
    parser.add_argument("--output_path", type=str, default="./results/mp_docvqa_100_results_Multi_Image_Concatenation.json")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    main(args)
