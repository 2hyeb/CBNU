path = '/home/user/CBNU/04_ragas_eval.py'
txt = open(path, encoding='utf-8').read()
changes = 0

# 1. tensor-parallel-size 4 추가 + max-model-len 8192
old1 = (
    '        "--gpu-memory-utilization", "0.90",\n'
    '        "--trust-remote-code",\n'
    '        "--max-model-len", "4096",\n'
)
new1 = (
    '        "--tensor-parallel-size", "4",\n'
    '        "--gpu-memory-utilization", "0.90",\n'
    '        "--trust-remote-code",\n'
    '        "--max-model-len", "8192",\n'
)
assert old1 in txt, "PATCH1 not found"
txt = txt.replace(old1, new1, 1); changes += 1

# 2. max_tokens 500 -> 2048
old2 = '                max_tokens=500,'
new2 = '                max_tokens=2048,'
assert old2 in txt, "PATCH2 not found"
txt = txt.replace(old2, new2, 1); changes += 1

# 3. MAX_CTX_CHARS 1500 -> 4000
old3 = 'MAX_CTX_CHARS = 1500   # truncate context to avoid 4096-token overflow'
new3 = 'MAX_CTX_CHARS = 4000   # truncate context (budget: 8192 token limit)'
assert old3 in txt, "PATCH3 not found"
txt = txt.replace(old3, new3, 1); changes += 1

# 4. MAX_ANS_CHARS 1000 -> 2500
old4 = 'MAX_ANS_CHARS = 1000   # truncate answer to avoid 4096-token overflow'
new4 = 'MAX_ANS_CHARS = 2500   # truncate answer (budget: 8192 token limit)'
assert old4 in txt, "PATCH4 not found"
txt = txt.replace(old4, new4, 1); changes += 1

open(path, 'w', encoding='utf-8').write(txt)
print(f'All {changes} patches applied OK')
