#!/bin/bash
# run_retrain.sh
# ft_dataset_rag.json으로 3모델 순차 재학습
# 실행: bash run_retrain.sh > logs/retrain.log 2>&1

set -e
cd /home/user/CBNU

PYTHON=/home/user/miniconda3/envs/vllm/bin/python
TRAIN_SCRIPT=06_train_qlora_v2.py
DATA_PATH=data/ft_dataset_rag.json
LOG_DIR=logs

mkdir -p $LOG_DIR

echo "====================================="
echo "재학습 시작: $(date)"
echo "데이터: $DATA_PATH"
echo "====================================="

for MODEL in exaone llama qwen3; do
    echo ""
    echo ">>> [$MODEL] 학습 시작: $(date)"

    # 기존 체크포인트 백업 (있으면)
    if [ -d "checkpoints/${MODEL}-ft" ]; then
        mv "checkpoints/${MODEL}-ft" "checkpoints/${MODEL}-ft-bak-$(date +%H%M)" || true
    fi
    if [ -d "checkpoints/${MODEL}-ft-merged" ]; then
        mv "checkpoints/${MODEL}-ft-merged" "checkpoints/${MODEL}-ft-merged-bak-$(date +%H%M)" || true
    fi

    # DATA_PATH를 ft_dataset_rag.json으로 덮어씌워서 실행
    # 06_train_qlora_v2.py의 DATA_PATH를 임시로 심볼릭 링크로 교체
    if [ -f "data/ft_dataset.json.orig_bak" ]; then
        : # already backed up
    else
        cp data/ft_dataset.json data/ft_dataset.json.orig_bak
    fi
    cp data/ft_dataset_rag.json data/ft_dataset.json

    $PYTHON $TRAIN_SCRIPT --model $MODEL 2>&1 | tee $LOG_DIR/retrain_${MODEL}.log

    echo ">>> [$MODEL] 학습 완료: $(date)"
done

# 원본 ft_dataset.json 복원
if [ -f "data/ft_dataset.json.orig_bak" ]; then
    cp data/ft_dataset.json.orig_bak data/ft_dataset.json
fi

echo ""
echo "====================================="
echo "전체 재학습 완료: $(date)"
echo "====================================="
