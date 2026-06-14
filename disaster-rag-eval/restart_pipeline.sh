#!/bin/bash
set -eo pipefail
cd /home/user/CBNU
mkdir -p logs

PYTHON_VLLM=/home/user/miniconda3/envs/vllm/bin/python
PYTHON_RAGAS=/home/user/miniconda3/envs/ragas_env/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [STEP A] 03_run_experiments_v2.py (8192 max_model_len) ==="
echo "시작: $(date)"
$PYTHON_RAGAS 03_run_experiments_v2.py 2>&1 | tee logs/restart_inference.log
echo "추론 완료: $(date)"

echo ""
echo "=== [STEP B] 남은 ERROR 항목 제거 ==="
python3 /home/user/CBNU/clean_errors.py

echo ""
echo "=== [STEP C] eval_summary에서 ft_rag + llama/rag 제거 ==="
$PYTHON_RAGAS -c "
import pandas as pd
from pathlib import Path
p = Path('data/eval_summary_by_type.csv')
df = pd.read_csv(p, encoding='utf-8-sig')
before = len(df)
mask = (df['condition'] == 'ft_rag') | ((df['model'] == 'llama') & (df['condition'] == 'rag'))
filtered = df[~mask]
removed = before - len(filtered)
print(f'eval_summary: {before} -> {len(filtered)} ({removed}개 제거)')
filtered.to_csv(p, index=False, encoding='utf-8-sig')
"

echo ""
echo "=== [STEP D] RAGAS 평가 ==="
echo "시작: $(date)"
$PYTHON_RAGAS 04_ragas_eval.py 2>&1 | tee logs/ragas_restart.log
echo "평가 완료: $(date)"

echo ""
echo "=== 최종 결과 ==="
$PYTHON_RAGAS -c "
import pandas as pd
df = pd.read_csv('data/eval_summary_by_type.csv', encoding='utf-8-sig')
print(df.to_string(index=False))
"
