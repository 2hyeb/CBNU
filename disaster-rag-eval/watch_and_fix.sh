#!/bin/bash
# watch_and_fix.sh
cd /home/user/CBNU
mkdir -p logs

echo "[$(date '+%H:%M:%S')] watcher started, waiting for PID 2962789 (03_run_experiments_v2.py)..."

# Wait for 03_run_experiments_v2.py to finish
while kill -0 2962789 2>/dev/null; do
    sleep 10
done
echo "[$(date '+%H:%M:%S')] 03_run_experiments_v2.py finished"

# Wait for 04_ragas_eval.py to start and finish (retrain_llama_qwen3.sh step 4)
sleep 5
if pgrep -f "04_ragas_eval" > /dev/null; then
    echo "[$(date '+%H:%M:%S')] 04_ragas_eval.py running, waiting..."
    while pgrep -f "04_ragas_eval" > /dev/null; do
        sleep 10
    done
    echo "[$(date '+%H:%M:%S')] 04_ragas_eval.py finished"
else
    echo "[$(date '+%H:%M:%S')] 04_ragas_eval.py not running yet, waiting 60s..."
    sleep 60
    while pgrep -f "04_ragas_eval" > /dev/null; do
        sleep 10
    done
    echo "[$(date '+%H:%M:%S')] 04_ragas_eval.py done"
fi

echo "[$(date '+%H:%M:%S')] Starting fix_llama_rerun.sh..."
bash /home/user/CBNU/fix_llama_rerun.sh 2>&1 | tee logs/fix_llama_rerun.log
echo "[$(date '+%H:%M:%S')] All done!"

