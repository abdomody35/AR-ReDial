import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

## set ENV variables
os.environ["HUGGINGFACE_API_KEY"] = "your-api-key"
os.environ["GEMINI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# pip install litellm
try:
    from litellm import completion
except ImportError:
    print("[ERROR] litellm not found. Please install it using 'pip install litellm'")

def extract_answer(response_text):
    """
    Extracts the content between <answer> and </answer> tags.
    """
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def format_prompt(item, task_type, dialect):
    """
    Combines dataset fields into a single prompt string based on task type.
    """
    premises = {
        "en" : "Consider the following premises:",
        "msa" : "ضع في اعتبارك الفرضيات التالية:",
        "egy" : "حط الفرضيات دي في بالك:",
        "jor" : "خلي الفرضيات دي في بالك:"
    }

    goal = {
        "en" : "Goal:",
        "msa" : "الهدف:",
        "egy" : "الهدف:",
        "jor" : "الهدف:"
    }

    steps = {
        "en" : "Steps:",
        "msa" : "الخطوات:",
        "egy" : "الخطوات:",
        "jor" : "الخطوات:"
    }

    constraints = {
        "en" : "Constraints:",
        "msa" : "القيود:",
        "egy" : "القيود:",
        "jor" : "القيود:"
    }

    question = {
        "en" : "Question:",
        "msa" : "السؤال:",
        "egy" : "السؤال:",
        "jor" : "السؤال:"
    }

    if task_type == 'logic':
        return f"{premises[dialect]}:\n{item.get('premises', '')}\n\n{question[dialect]}\n{item.get('question', '')}"
    elif task_type == 'math':
        return item.get('question', '')
    elif task_type == 'reasoning':
        return f"{goal[dialect]} {item.get('goal', '')}\n{steps[dialect]}\n{item.get('steps', '')}\n{constraints[dialect]}\n{item.get('constraints', '')}\n\n{question[dialect]}\n{item.get('question', '')}"
    return item.get('question', 'No question found.')

def evaluate_file(model_name, data_file, results_dir):
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    filename = data_file.stem
    if 'logic' in filename: task_type = 'logic'
    elif 'math' in filename: task_type = 'math'
    elif 'reasoning' in filename: task_type = 'reasoning'
    else: task_type = 'unknown'
    
    dialect = data_file.parent.name
    
    results_path = Path(results_dir) / model_name.replace("/", "_") / dialect
    results_path.mkdir(parents=True, exist_ok=True)
    output_file = results_path / f"{data_file.name}"
    
    print(f"\n[START] Evaluating Model: {model_name}")
    print(f"[INFO] Task: {task_type} | Dialect: {dialect} | File: {data_file.name}")
    print(f"[INFO] Number of items to process: {len(data)}")
    
    results = []
    for item in tqdm(data, desc=f"Processing {dialect}/{task_type}"):
        prompt = format_prompt(item, task_type, dialect)
        item_id = str(item.get('id'))
        
        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            
            model_raw_output = response.choices[0].message.content
            model_extracted_answer = extract_answer(model_raw_output)
            
            results.append({
                "id": item_id,
                "prompt": prompt,
                "raw_output": model_raw_output,
                "extracted_answer": model_extracted_answer,
                "correct_answer": item.get('answer')
            })
            
        except Exception as e:
            print(f"\n[ERROR] Failed on ID {item_id}: {e}")
            results.append({
                "id": item_id,
                "prompt": prompt,
                "error": str(e),
                "correct_answer": item.get('answer')
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on datasets.")
    parser.add_argument("--model", type=str, required=True, help="LiteLLM model name")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save output JSONs")
    
    args = parser.parse_args()
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        print(f"[ERROR] Data directory '{args.data_dir}' not found.")
        return

    print(f"--- Evaluation Session Started ---")
    print(f"Model: {args.model}")
    print(f"Data Source: {args.data_dir}")
    print(f"Results Dest: {args.results_dir}")

    # Iterate through language folders
    dialect_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    for dialect_dir in dialect_dirs:
        json_files = sorted(list(dialect_dir.glob("*.json")))
        for json_file in json_files:
            evaluate_file(args.model, json_file, args.results_dir)
            
    print(f"\n--- Evaluation Session Complete ---")

if __name__ == "__main__":
    main()
