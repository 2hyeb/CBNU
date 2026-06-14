#!/bin/bash
# retrain_llama_qwen3.sh
# LLaMA + Qwen3 ??? (GPU 0+1+2+3) ? ft_rag ??? + RAGAS ??
set -eo pipefail
cd /home/user/CBNU
mkdir -p logs

PYTHON_VLLM=/home/user/miniconda3/envs/vllm/bin/python
PYTHON_RAGAS=/home/user/miniconda3/envs/ragas_env/bin/python
export LD_LIBRARY_PATH=/home/user/miniconda3/envs/ragas_env/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "==========================================="
echo "[STEP 2b] LLaMA + Qwen3 ??? (GPU 0+1+2+3)"
echo "??: $(date)"
echo "==========================================="

for MODEL in llama qwen3; do
    echo ""
    echo ">>> [$MODEL] ?? ??: $(date)"

    if [ -d "checkpoints/${MODEL}-ft" ]; then
        mv "checkpoints/${MODEL}-ft" "checkpoints/${MODEL}-ft-old-$(date +%H%M)"
    fi
    if [ -d "checkpoints/${MODEL}-ft-merged" ]; then
        mv "checkpoints/${MODEL}-ft-merged" "checkpoints/${MODEL}-ft-merged-old-$(date +%H%M)"
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON_VLLM 06_train_qlora_v2.py --model $MODEL 2>&1 | tee logs/retrain_${MODEL}.log
    echo ">>> [$MODEL] ??: $(date)"
done

# ?? ??? ??
cp data/ft_dataset.json.orig_bak data/ft_dataset.json
echo "?? ft_dataset.json ?? ??"

echo ""
echo "==========================================="
echo "[STEP 3] ft_rag ???"
echo "??: $(date)"
echo "==========================================="

$PYTHON_RAGAS -c "
import json
from pathlib import Path
p = Path('data/experiment_results_v2.json')
data = json.loads(p.read_text(encoding='utf-8'))
filtered = [r for r in data if r.get('condition') != 'ft_rag']
removed = len(data) - len(filtered)
print(f'experiment_results: {len(data)} -> {len(filtered)} (ft_rag {removed}? ??)')
p.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding='utf-8')
"

echo "03_run_experiments_v2.py ?? (ft_rag? ???)"
$PYTHON_RAGAS 03_run_experiments_v2.py 2>&1 | tee logs/run_ftrag.log
echo "?? ??: $(date)"

echo ""
echo "==========================================="
echo "[STEP 4] ft_rag RAGAS ??"
echo "??: $(date)"
echo "==========================================="

$PYTHON_RAGAS -c "
import pandas as pd
from pathlib import Path
p = Path('data/eval_summary_by_type.csv')
df = pd.read_csv(p, encoding='utf-8-sig')
filtered = df[df['condition'] != 'ft_rag']
removed = len(df) - len(filtered)
print(f'eval_summary: {len(df)} -> {len(filtered)} (ft_rag {removed}? ??)')
filtered.to_csv(p, index=False, encoding='utf-8-sig')
"

echo "04_ragas_eval.py ?? (ft_rag? ???)"
$PYTHON_RAGAS 04_ragas_eval.py 2>&1 | tee logs/ragas_ftrag.log

echo ""
echo "==========================================="
echo "?? ??: $(date)"
echo "?? ??: data/eval_summary_by_type.csv"
echo "==========================================="
$PYTHON_RAGAS -c "
import pandas as pd
df = pd.read_csv('data/eval_summary_by_type.csv', encoding='utf-8-sig')
print(df.to_string(index=False))
"
