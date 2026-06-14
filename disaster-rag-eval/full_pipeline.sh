#!/bin/bash
# full_pipeline.sh
# ft_rag 수정 전체 파이프라인:
# 1) ft_dataset 변환 (ragas_env, FAISS context 추가)
# 2) 3모델 재학습 (vllm env)
# 3) ft_rag 결과만 삭제 → 재실험 (ragas_env)
# 4) ft_rag RAGAS 평가 (ragas_env)

set -eo pipefail
cd /home/user/CBNU
mkdir -p logs

PYTHON_VLLM=/home/user/miniconda3/envs/vllm/bin/python
PYTHON_RAGAS=/home/user/miniconda3/envs/ragas_env/bin/python
export LD_LIBRARY_PATH=/home/user/miniconda3/envs/ragas_env/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "==========================================="
echo "[STEP 1] ft_dataset 변환 (ragas_env)"
echo "시작: $(date)"
echo "==========================================="

# 이미 생성된 경우 스킵
if [ -f data/ft_dataset_rag.json ]; then
    echo "ft_dataset_rag.json 이미 존재 — 스킵"
else
    $PYTHON_RAGAS convert_ft_dataset.py 2>&1 | tee logs/convert_ft.log
fi
echo "변환 완료: $(date)"

echo ""
echo "==========================================="
echo "[STEP 2] 3모델 재학습 (vllm env)"
echo "시작: $(date)"
echo "==========================================="

# 학습 데이터 교체
cp data/ft_dataset.json data/ft_dataset.json.orig_bak
cp data/ft_dataset_rag.json data/ft_dataset.json
echo "데이터 교체 완료: ft_dataset_rag.json → ft_dataset.json"

for MODEL in exaone llama qwen3; do
    echo ""
    echo ">>> [$MODEL] 학습 시작: $(date)"

    # 기존 체크포인트 백업
    if [ -d "checkpoints/${MODEL}-ft" ]; then
        mv "checkpoints/${MODEL}-ft" "checkpoints/${MODEL}-ft-old-$(date +%H%M)"
    fi
    if [ -d "checkpoints/${MODEL}-ft-merged" ]; then
        mv "checkpoints/${MODEL}-ft-merged" "checkpoints/${MODEL}-ft-merged-old-$(date +%H%M)"
    fi

    $PYTHON_VLLM 06_train_qlora_v2.py --model $MODEL 2>&1 | tee logs/retrain_${MODEL}.log
    echo ">>> [$MODEL] 완료: $(date)"
done

# 원본 데이터 복원
cp data/ft_dataset.json.orig_bak data/ft_dataset.json
echo "원본 ft_dataset.json 복원 완료"

echo ""
echo "==========================================="
echo "[STEP 3] ft_rag 기존 결과 제거 + 재실험"
echo "시작: $(date)"
echo "==========================================="

# experiment_results_v2.json에서 ft_rag 조건만 제거
$PYTHON_RAGAS -c "
import json
from pathlib import Path
p = Path('data/experiment_results_v2.json')
data = json.loads(p.read_text(encoding='utf-8'))
filtered = [r for r in data if r.get('condition') != 'ft_rag']
removed = len(data) - len(filtered)
print(f'experiment_results: {len(data)} → {len(filtered)} (ft_rag {removed}개 제거)')
p.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding='utf-8')
"

# ft_rag 조건 재실험 (done_keys 덕에 ft_rag만 실행됨)
echo "03_run_experiments_v2.py 실행 (ft_rag만 처리됨)"
$PYTHON_RAGAS 03_run_experiments_v2.py 2>&1 | tee logs/run_ftrag.log
echo "실험 완료: $(date)"

echo ""
echo "==========================================="
echo "[STEP 4] ft_rag RAGAS 평가"
echo "시작: $(date)"
echo "==========================================="

# eval_summary_by_type.csv에서 ft_rag 행 제거 (재평가 대상)
$PYTHON_RAGAS -c "
import pandas as pd
from pathlib import Path
p = Path('data/eval_summary_by_type.csv')
df = pd.read_csv(p, encoding='utf-8-sig')
filtered = df[df['condition'] != 'ft_rag']
removed = len(df) - len(filtered)
print(f'eval_summary: {len(df)} → {len(filtered)} (ft_rag {removed}개 제거)')
filtered.to_csv(p, index=False, encoding='utf-8-sig')
"

# RAGAS 평가 (done_keys 덕에 ft_rag만 처리됨)
echo "04_ragas_eval.py 실행 (ft_rag만 평가됨)"
$PYTHON_RAGAS 04_ragas_eval.py 2>&1 | tee logs/ragas_ftrag.log

echo ""
echo "==========================================="
echo "전체 완료: $(date)"
echo "최종 결과: data/eval_summary_by_type.csv"
echo "==========================================="
$PYTHON_RAGAS -c "
import pandas as pd
df = pd.read_csv('data/eval_summary_by_type.csv', encoding='utf-8-sig')
print(df.to_string(index=False))
"
