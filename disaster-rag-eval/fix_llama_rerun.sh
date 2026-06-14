#!/bin/bash
# fix_llama_rerun.sh  (updated: fix ALL ft_rag + llama rag)
# max_model_len: 2048 -> 4096 already patched in 03_run_experiments_v2.py
set -eo pipefail
cd /home/user/CBNU
mkdir -p logs

PYTHON_VLLM=/home/user/miniconda3/envs/vllm/bin/python
PYTHON_RAGAS=/home/user/miniconda3/envs/ragas_env/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [1] ERROR 항목 제거 (exaone/llama/qwen3 ft_rag + llama rag) ==="
$PYTHON_RAGAS -c "
import json
from pathlib import Path
p = Path('data/experiment_results_v2.json')
d = json.loads(p.read_text(encoding='utf-8'))
before = len(d)
def is_error(r):
    bad_conds = (
        (r.get('condition') == 'ft_rag') or
        (r.get('model') == 'llama' and r.get('condition') == 'rag')
    )
    return bad_conds and str(r.get('answer', '')).startswith('ERROR')
filtered = [r for r in d if not is_error(r)]
removed = before - len(filtered)
print(f'ERROR 제거: {removed}개 | 잔여: {len(filtered)}개')
p.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding='utf-8')
"

echo ""
echo "=== [2] 03_run_experiments_v2.py 재실행 (done_keys 기반, 누락분만 처리) ==="
$PYTHON_RAGAS 03_run_experiments_v2.py 2>&1 | tee logs/rerun_all_ftrag.log

echo ""

echo ""
echo "=== [2b] exaone ft_rag ERROR 재처리 ==="
python3 -c "
import json; p='data/experiment_results_v2.json'
d=json.load(open(p))
bad=[r for r in d if r.get('model')=='exaone' and r.get('condition')=='ft_rag' and str(r.get('answer','')).startswith('ERROR')]
cleaned=[r for r in d if r not in bad]
print(f'exaone ft_rag error removed: {len(bad)}')
json.dump(cleaned, open(p,'w'), ensure_ascii=False, indent=2)
"
$PYTHON_RAGAS 03_run_experiments_v2.py 2>&1 | tee -a logs/rerun_all_ftrag.log

echo "=== [3] eval_summary에서 재평가 대상 행 제거 ==="
$PYTHON_RAGAS -c "
import pandas as pd
from pathlib import Path
p = Path('data/eval_summary_by_type.csv')
df = pd.read_csv(p, encoding='utf-8-sig')
before = len(df)
# ft_rag 전체 + llama rag 제거
mask = (df['condition'] == 'ft_rag') | ((df['model'] == 'llama') & (df['condition'] == 'rag'))
filtered = df[~mask]
removed = before - len(filtered)
print(f'eval_summary: {before} -> {len(filtered)} ({removed}개 제거)')
filtered.to_csv(p, index=False, encoding='utf-8-sig')
"

echo ""
echo "=== [4] RAGAS 재평가 (ft_rag + llama rag) ==="
$PYTHON_RAGAS 04_ragas_eval.py 2>&1 | tee logs/ragas_final.log

echo ""
echo "=== 최종 결과 ==="
$PYTHON_RAGAS -c "
import pandas as pd
df = pd.read_csv('data/eval_summary_by_type.csv', encoding='utf-8-sig')
print(df.to_string(index=False))
"

