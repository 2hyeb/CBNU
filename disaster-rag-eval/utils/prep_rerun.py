# -*- coding: utf-8 -*-
"""Remove llama entries from raw JSON so the eval script re-runs them as llama31."""
import json
from pathlib import Path

RAW = Path('/home/user/CBNU/data/eval_qtype_raw.json')

data = json.loads(RAW.read_text(encoding='utf-8'))
print(f'Before: {len(data)} entries')
print(f'  llama entries: {sum(1 for d in data if d.get("model")=="llama")}')
print(f'  others: {sum(1 for d in data if d.get("model")!="llama")}')

# Also check llama31 in experiment_results
EXP = Path('/home/user/CBNU/data/experiment_results_v2.json')
exp = json.loads(EXP.read_text(encoding='utf-8'))
llama31_rag = [d for d in exp if d.get('model')=='llama31' and d.get('condition')=='rag']
llama31_ft  = [d for d in exp if d.get('model')=='llama31' and d.get('condition')=='ft_only']
print(f'\nllama31/rag in experiments: {len(llama31_rag)}')
print(f'llama31/ft_only in experiments: {len(llama31_ft)}')
if llama31_ft:
    sample = llama31_ft[0]
    ans = sample.get('pred_answer','')
    print(f'  ft_only sample answer len={len(str(ans))} val={repr(str(ans))[:100]}')

# Keep only non-llama entries
kept = [d for d in data if d.get('model') != 'llama']
print(f'\nAfter removal: {len(kept)} entries kept')

RAW.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved. Ready for llama31 re-run.')
