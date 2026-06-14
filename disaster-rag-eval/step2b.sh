
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

