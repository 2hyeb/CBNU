import json
from pathlib import Path
from transformers import AutoTokenizer

data = json.loads(Path('data/ft_dataset_rag.json').read_text())
tok = AutoTokenizer.from_pretrained('/home/user/models/EXAONE-3.5-7.8B-Instruct', trust_remote_code=True)
lengths = []
for item in data:
    text = tok.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=False)
    lengths.append(len(tok(text)['input_ids']))
srt = sorted(lengths)
print(f'n={len(lengths)} min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)} p90={srt[int(len(lengths)*0.90)]} p95={srt[int(len(lengths)*0.95)]}')
