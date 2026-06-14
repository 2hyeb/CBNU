# -*- coding: utf-8 -*-
import json

with open('/home/user/CBNU/data/experiment_results_v2.json', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total entries: {len(data)}')
print('Sample keys:', list(data[0].keys()) if data else 'EMPTY')
print()

# model 필드 유니크 값
models = set(d.get('model','') for d in data)
conds  = set(d.get('condition','') for d in data)
print('Models:', models)
print('Conditions:', conds)
print()

# LLaMA/rag 찾기 (model 필드 이름 유연하게)
llama_keys = [m for m in models if 'llama' in m.lower() or 'Llama' in m]
print('LLaMA model keys:', llama_keys)

for lk in llama_keys:
    items = [d for d in data if d.get('model')==lk and d.get('condition')=='rag']
    print(f'\n{lk}/rag count: {len(items)}')
    if items:
        s = items[0]
        ans = s.get('pred_answer','')
        ctx = s.get('context','')
        print(f'  pred_answer len={len(str(ans))}  val={repr(str(ans))[:150]}')
        print(f'  context len={len(str(ctx))}  val={repr(str(ctx))[:150]}')

        empty_ans = sum(1 for d in items if not str(d.get('pred_answer','')).strip() or str(d.get('pred_answer','')).startswith('ERROR'))
        empty_ctx = sum(1 for d in items if not str(d.get('context','')).strip())
        print(f'  empty/ERROR pred_answer: {empty_ans}/{len(items)}')
        print(f'  empty context: {empty_ctx}/{len(items)}')

        # 비어있지 않은 샘플
        non_empty = [d for d in items if str(d.get('pred_answer','')).strip() and not str(d.get('pred_answer','')).startswith('ERROR')]
        if non_empty:
            print(f'\n  Non-empty answer sample:')
            print(f'  Q: {non_empty[0].get("question","")[:80]}')
            print(f'  A: {repr(non_empty[0].get("pred_answer",""))[:300]}')
            print(f'  ctx: {repr(non_empty[0].get("context",""))[:200]}')
        else:
            print('  ALL pred_answers empty or ERROR')
