import json
from pathlib import Path
p = Path('/home/user/CBNU/data/experiment_results_v2.json')
d = json.loads(p.read_text(encoding='utf-8'))
before = len(d)
cleaned = [r for r in d if not str(r.get('answer','')).startswith('ERROR')]
removed = before - len(cleaned)
print(f'ERROR 제거: {removed}개 | before: {before} -> after: {len(cleaned)}')
# summary
from collections import defaultdict
s = defaultdict(lambda: [0,0])
for r in cleaned:
    k = r.get('model','?')+'/'+r.get('condition','?')
    s[k][0] += 1
for k in sorted(s):
    print(f'  {k}: {s[k][0]}')
p.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding='utf-8')
print('Done.')
