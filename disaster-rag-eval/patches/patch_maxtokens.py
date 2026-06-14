path = '/home/user/CBNU/04_ragas_eval.py'
txt = open(path, encoding='utf-8').read()
old = 'max_tokens=200,'
new = 'max_tokens=500,'
assert old in txt, 'target not found: ' + repr(old)
open(path, 'w', encoding='utf-8').write(txt.replace(old, new, 1))
print('max_tokens 200 -> 500 OK')
