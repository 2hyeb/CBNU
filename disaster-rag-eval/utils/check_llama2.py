# -*- coding: utf-8 -*-
import json, math

with open('/home/user/CBNU/data/eval_qtype_raw.json') as f:
    data = json.load(f)

llama_rag = [d for d in data if d.get('model')=='llama' and d.get('condition')=='rag']

print('=== contexts field check (first 5) ===')
for d in llama_rag[:5]:
    ctx = d.get('contexts')
    print(f'  type={type(ctx).__name__}  val={repr(ctx)[:100]}')

print()
print('=== answer field check (first 5) ===')
for d in llama_rag[:5]:
    ans = d.get('answer')
    print(f'  type={type(ans).__name__}  val={repr(ans)[:120]}')

print()
empty_ans = sum(1 for d in llama_rag if not d.get('answer','').strip())
empty_ctx = sum(1 for d in llama_rag if not d.get('contexts'))
print(f'empty answers: {empty_ans}/{len(llama_rag)}')
print(f'empty contexts: {empty_ctx}/{len(llama_rag)}')

print()
print('=== keys in one entry ===')
print(list(llama_rag[0].keys()))

print()
print('=== non-empty answer sample ===')
non_empty = [d for d in llama_rag if d.get('answer','').strip()]
if non_empty:
    for d in non_empty[:2]:
        print(repr(d.get('answer',''))[:300])
        print()
else:
    print('ALL EMPTY')
