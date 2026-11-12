"""Run local evaluation against data/dev.jsonl using score.py.

Usage:
    python run_evaluation.py
"""

import json, os, subprocess
import generate_test_predictions
import score

def main():
    print("Evaluating triage model on clean dev set (data/dev.jsonl)...")
    dev_rows = [json.loads(line) for line in open('data/dev.jsonl', encoding='utf-8')]
    
    preds = []
    for d in dev_rows:
        pred_obj = generate_test_predictions.predict(d['messages'][1]['content'])
        preds.append({
            'id': d['id'],
            'output': json.dumps(pred_obj)
        })
        
    temp_preds_path = 'temp_dev_preds.jsonl'
    with open(temp_preds_path, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps(p) + '\n')
            
    res = score.score(dev_rows, preds)
    
    if os.path.exists(temp_preds_path):
        os.remove(temp_preds_path)
        
    print("\n" + "=" * 65)
    print("           NIMBUSPAY TRIAGE EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Dev Rows:          {res['all']['n']}")
    print(f"JSON Validity Rate:      {res['all']['json_valid']:.1%}")
    print(f"Dev Exact Match:         {res['all']['exact_match']:.1%}")
    print(f"Dev Mean Field Accuracy: {res['all']['mean_field_acc']:.1%}")
    print("-" * 65)
    print("Per-Field Accuracy Breakdown:")
    for k, v in res['all']['fields'].items():
        print(f"  {k:16} : {v:6.1%}")
    print("=" * 65)

if __name__ == '__main__':
    main()
