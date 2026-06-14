path = '/home/user/CBNU/04_ragas_eval.py'
txt = open(path, encoding='utf-8').read()

old = '        "--tensor-parallel-size", "4",\n'
new = '        "--tensor-parallel-size", "2",\n'
assert old in txt, "PATCH not found"
txt = txt.replace(old, new, 1)

open(path, 'w', encoding='utf-8').write(txt)
print('tensor-parallel-size 4 -> 2 OK')
print('Verify:', [l.strip() for l in txt.splitlines() if 'tensor-parallel' in l])
