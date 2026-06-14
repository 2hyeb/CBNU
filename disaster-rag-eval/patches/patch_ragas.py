import re

path = '/home/user/CBNU/04_ragas_eval.py'
txt  = open(path, encoding='utf-8').read()

# 1. Add MAX_CTX/ANS constants after CONDITIONS line
old1 = 'CONDITIONS = ["no_rag", "rag", "ft_only", "ft_rag"]'
new1 = ('CONDITIONS = ["no_rag", "rag", "ft_only", "ft_rag"]\n\n'
        'MAX_CTX_CHARS = 1500   # truncate context to avoid 4096-token overflow\n'
        'MAX_ANS_CHARS = 1000   # truncate answer to avoid 4096-token overflow')
assert old1 in txt, "PATCH1 target not found"
txt = txt.replace(old1, new1, 1)

# 2. Add resume/skip logic after "summary = []" before "for model in MODELS:"
old2 = '        summary = []\n        for model in MODELS:'
new2 = (
    '        summary = []\n'
    '        done_keys = set()\n'
    '        if SUMMARY_PATH.exists():\n'
    '            try:\n'
    '                df_ex = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")\n'
    '                for _, row in df_ex.iterrows():\n'
    '                    has_ctx  = row["condition"] in ("rag", "ft_rag")\n'
    '                    ans_ok   = pd.notna(row.get("answer_relevancy"))\n'
    '                    faith_ok = pd.notna(row.get("faithfulness"))  if has_ctx else True\n'
    '                    ctx_ok   = pd.notna(row.get("context_recall")) if has_ctx else True\n'
    '                    if ans_ok and faith_ok and ctx_ok:\n'
    '                        done_keys.add((row["model"], row["condition"]))\n'
    '                        summary.append(row.to_dict())\n'
    '                        print(f"[RESUME] {row[\'model\']}/{row[\'condition\']} — already valid, skipping")\n'
    '            except Exception as e:\n'
    '                print(f"Could not read existing CSV: {e}")\n'
    '\n'
    '        for model in MODELS:'
)
assert old2 in txt, "PATCH2 target not found"
txt = txt.replace(old2, new2, 1)

# 3. Add done_keys skip check after the no-data skip block
old3 = ('                if not subset:\n'
        '                    print(f"[SKIP] {model}/{cond} — no data")\n'
        '                    continue')
new3 = (
    '                if not subset:\n'
    '                    print(f"[SKIP] {model}/{cond} — no data")\n'
    '                    continue\n'
    '\n'
    '                if (model, cond) in done_keys:\n'
    '                    print(f"[SKIP] {model}/{cond} — already valid in CSV")\n'
    '                    continue'
)
assert old3 in txt, "PATCH3 target not found"
txt = txt.replace(old3, new3, 1)

# 4. Truncate answer in ds_dict
old4 = '                    "answer":       [r.get("answer") or r.get("pred_answer", "") for r in subset],'
new4 = '                    "answer":       [(r.get("answer") or r.get("pred_answer", ""))[:MAX_ANS_CHARS] for r in subset],'
assert old4 in txt, "PATCH4 target not found"
txt = txt.replace(old4, new4, 1)

# 5. Truncate context in ds_dict
old5 = '                    "contexts":     [[r.get("context", "")] for r in subset],'
new5 = '                    "contexts":     [[r.get("context", "")[:MAX_CTX_CHARS]] for r in subset],'
assert old5 in txt, "PATCH5 target not found"
txt = txt.replace(old5, new5, 1)

open(path, 'w', encoding='utf-8').write(txt)
print('All 5 patches applied OK')
