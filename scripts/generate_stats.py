import json
import re
import argparse
import csv
from pathlib import Path

def check_correctness(task_type, gold, model_ans):
    """
    Performs task-specific correctness checking.
    Returns 1 if correct, 0 otherwise.
    """
    if not model_ans: return 0
    gold = str(gold).strip().lower()
    model_ans = str(model_ans).strip().lower()
    
    if task_type == 'logic':
        return 1 if gold in model_ans or model_ans in gold else 0
    
    if task_type == 'math':
        try:
            g_num = float(re.sub(r'[^\d.]', '', gold))
            m_num = float(re.sub(r'[^\d.]', '', model_ans))
            return 1 if abs(g_num - m_num) < 1e-6 else 0
        except:
            return 1 if gold in model_ans else 0

    if task_type == 'reasoning':
        try:
            def to_secs_gold(s):
                d = int(re.search(r'days=(\d+)', s).group(1)) if 'days=' in s else 0
                s_val = int(re.search(r'seconds=(\d+)', s).group(1)) if 'seconds=' in s else 0
                return d * 86400 + s_val
            
            gold_secs = to_secs_gold(gold)
            units = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400, 'week': 604800, 'month': 2592000}
            parts = re.findall(r'(\d+)\s*(sec|min|hour|day|week|month)s?', model_ans)
            model_secs = sum(int(v) * units[u] for v, u in parts)
            return 1 if gold_secs == model_secs and model_secs > 0 else 0
        except:
            return 0
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate SPSS CSVs from result JSONs.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory for a specific model (e.g., results/gemini-pro)")
    args = parser.parse_args()
    
    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"[ERROR] Results directory '{args.results_dir}' not found.")
        return

    print(f"--- Statistics Generation Started ---")
    print(f"[INFO] Scanning directory: {results_root}")

    # stats[task_type][id][dialect] = 0/1
    stats = {}
    dialects = ['en', 'msa', 'egy', 'jor']

    # Traverse dialect folders
    dialect_dirs = sorted([d for d in results_root.iterdir() if d.is_dir() and d.name in dialects])
    
    if not dialect_dirs:
        print(f"[WARN] No valid dialect subdirectories (en, msa, egy, jor) found in {results_root}")

    for dialect_dir in dialect_dirs:
        dialect = dialect_dir.name
        print(f"\n[FOLDER] Processing dialect: {dialect}")
        
        json_files = sorted(list(dialect_dir.glob("*.json")))
        for json_file in json_files:
            fn = json_file.stem
            if 'logic' in fn: task_type = 'logic'
            elif 'math' in fn: task_type = 'math'
            elif 'reasoning' in fn: task_type = 'reasoning'
            else: 
                print(f"  [SKIP] Unknown task type: {json_file.name}")
                continue
            
            print(f"  [FILE] Scoring {task_type}...")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if task_type not in stats: stats[task_type] = {}
            
            correct_count = 0
            for item in data:
                iid = str(item.get('id'))
                if iid not in stats[task_type]: stats[task_type][iid] = {}
                
                model_ans = item.get('extracted_answer', item.get('raw_output', ''))
                gold = item.get('correct_answer', '')
                
                is_correct = check_correctness(task_type, gold, model_ans)
                stats[task_type][iid][dialect] = is_correct
                if is_correct: correct_count += 1
            
            accuracy = (correct_count / len(data)) * 100 if data else 0
            print(f"  [RESULT] {dialect}/{task_type}: {correct_count}/{len(data)} correct ({accuracy:.2f}%)")

    print(f"\n[FINALIZE] Generating SPSS CSVs...")
    for task_type, items in stats.items():
        csv_file = results_root / f"{task_type}_results.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id'] + dialects)
            
            sorted_ids = sorted(items.keys(), key=lambda x: int(x) if x.isdigit() else x)
            for iid in sorted_ids:
                row = [iid]
                for d in dialects:
                    row.append(items[iid].get(d, 0))
                writer.writerow(row)
        print(f"[SUCCESS] Created: {csv_file}")

    print(f"\n--- Statistics Generation Complete ---")

if __name__ == "__main__":
    main()
