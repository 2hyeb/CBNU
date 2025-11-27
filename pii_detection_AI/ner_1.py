import re
import torch
from config import ID_TAG
from modules.model_loader import ModelLoader

_loader = ModelLoader()
tokenizer = _loader.tokenizer
model = _loader.model
device = _loader.device

def ner_inference(text):
    """1차 추론"""
    text = text.replace(' ', '_')
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    batch = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', torch.zeros_like(batch['input_ids']))
        )
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    max_probs, preds = torch.max(probs, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
    pred_tags = [ID_TAG[idx] for idx in preds[0].cpu().numpy().tolist()]
    pred_probs = [float(p) for p in max_probs[0].cpu().numpy().tolist()]
    return tokens, pred_tags, pred_probs


def is_valid_name(name: str) -> bool:
    hanja_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f]')
    if hanja_pattern.search(name):
        return False
    if name.isdigit():
        return False
    if all(char.isalpha() and char.isascii() for char in name):
        return False
    if len(name.strip()) <= 1:
        return False
    if any(char.isdigit() for char in name) and any(char.isalpha() for char in name):
        return False
    if any(not char.isalnum() for char in name):
        return False
    return True

def detect_names(tokens, pred_tags, pred_probs):
    """토큰, 태그, 확률 리스트를 받아 PER 엔티티 추출"""
    names = []
    current_entity = ''
    for token, tag, prob in zip(tokens, pred_tags, pred_probs):
        token_clean = token.replace('▁', '').replace('##', '')
        if tag == 'PER_B' and prob >= 0.6:
            if current_entity and is_valid_name(current_entity):
                names.append(current_entity)
            current_entity = token_clean
        elif tag == 'PER_I' and current_entity and prob >= 0.6:
            current_entity += token_clean
        else:
            if current_entity and is_valid_name(current_entity):
                names.append(current_entity)
            current_entity = ''
    if current_entity and is_valid_name(current_entity):
        names.append(current_entity)
    return names