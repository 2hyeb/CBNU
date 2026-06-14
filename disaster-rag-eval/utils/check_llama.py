# -*- coding: utf-8 -*-
import json, math

with open('/home/user/CBNU/data/eval_qtype_raw.json') as f:
    data = json.load(f)

llama_rag = [d for d in data if d.get('model')=='llama' and d.get('condition')=='rag']
print(f'LLaMA/rag total: {len(llama_rag)}')

faiths = [d.get('faithfulness') for d in llama_rag]
nan_c  = sum(1 for f in faiths if f is None or (isinstance(f,float) and math.isnan(f)))
zero_c = sum(1 for f in faiths if isinstance(f,float) and f == 0.0)
pos_c  = sum(1 for f in faiths if isinstance(f,float) and not math.isnan(f) and f > 0.0)
print(f'faithfulness NaN/None: {nan_c}')
print(f'faithfulness 0.0:      {zero_c}')
print(f'faithfulness >0:       {pos_c}')
print()

by_qtype = {}
for d in llama_rag:
    qt = d.get('q_type','?')
    f  = d.get('faithfulness')
    by_qtype.setdefault(qt, []).append(f)

for qt, vals in sorted(by_qtype.items()):
    non_nan = [v for v in vals if isinstance(v,float) and not math.isnan(v)]
    avg = sum(non_nan)/len(non_nan) if non_nan else float('nan')
    nan_n = sum(1 for v in vals if v is None or (isinstance(v,float) and math.isnan(v)))
    zero_n = sum(1 for v in non_nan if v==0.0)
    print(f'{qt}: n={len(vals)} avg={avg:.4f} nan={nan_n} zero={zero_n} pos={len(non_nan)-zero_n}')

print()
print('=== zero sample ===')
zero_s = [d for d in llama_rag if isinstance(d.get('faithfulness'),float) and d['faithfulness']==0.0]
for s in zero_s[:2]:
    print(f'q_type={s.get("q_type")}')
    print('Q:', s.get('question','')[:100])
    print('A:', s.get('answer','')[:400])
    print('ctx len:', len(str(s.get('contexts',''))))
    print()

print('=== nan/none sample ===')
nan_s = [d for d in llama_rag if d.get('faithfulness') is None or (isinstance(d.get('faithfulness'),float) and math.isnan(d['faithfulness']))]
for s in nan_s[:2]:
    print(f'q_type={s.get("q_type")}')
    print('Q:', s.get('question','')[:100])
    print('A:', s.get('answer','')[:400])
    print('ctx len:', len(str(s.get('contexts',''))))
    print()

print('=== pos sample ===')
pos_s = [d for d in llama_rag if isinstance(d.get('faithfulness'),float) and not math.isnan(d['faithfulness']) and d['faithfulness']>0]
for s in pos_s[:2]:
    print(f'q_type={s.get("q_type")} faith={s["faithfulness"]:.3f}')
    print('Q:', s.get('question','')[:100])
    print('A:', s.get('answer','')[:400])
    print()
