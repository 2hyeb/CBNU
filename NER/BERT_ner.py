import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer, AutoModelForTokenClassification, EarlyStoppingCallback
import sys

if torch.cuda.is_available():    
    device = torch.device("cuda")

    # print('There are %d GPU(s) available.' % torch.cuda.device_count())

    # print('GPU:', torch.cuda.get_device_name(0))
else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

data_df = pd.read_csv('train_data.csv', index_col = 0)

label = []

def find_label(labels):
    for i in labels:
        if i not in label:
            label.append(i)

data_df['tag'].str.split().apply(find_label)
label.sort()
label = ['UNK'] + label + ['POS'] + ['ADJ'] + ['SUF']

tag_id = {tag : i for tag, i in enumerate(label)}
id_tag = {i : tag for tag, i in tag_id.items()}

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, labels):
        # input_data를 딕셔너리로 변환
        self.encodings = {
            "input_ids": input_data[0],
            "attention_mask": input_data[1],
            "token_type_ids": input_data[2]
        }
        self.labels = labels

    def __getitem__(self, idx):
        # encodings 딕셔너리에서 idx에 해당하는 데이터를 추출
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = torch.load('train_dataset.pth', weights_only=False)
test_dataset = torch.load('test_dataset.pth', weights_only=False)

# 모델 불러오기
model = AutoModelForTokenClassification.from_pretrained('./BERT_base_2/checkpoint-10530', num_labels=len(tag_id))
model.to(device)

index_to_ner = {i: j for j, i in tag_id.items()}

index_to_ner = {v: k for k, v in index_to_ner.items()}

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#----------------------------------------------------------------------------------------------------------------------
def ner_inference(text):  
    model.eval()
    text = text.replace(' ', '_')

    predictions, true_labels = [], []
    probs = []  # 확률을 저장할 리스트

    #입력으로 받는 텍스트를 토큰화
    tokenized_sent = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tokenized_sent['input_ids'].to(device)
    attention_mask = tokenized_sent['attention_mask'].to(device)
    token_type_ids = tokenized_sent.get('token_type_ids', torch.zeros_like(input_ids)).to(device)

    with torch.no_grad(): #모델을 추론모드로 전환 (파라미터 업데이트 x)
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids)
    
    #logits 값 추출
    logits = outputs.logits.detach().cpu().numpy()  #기울기가 계산되지 않도록하고, logits 값을 CPU로 이동

    #softmax를 통해 확률 계산
    probs_array = torch.softmax(torch.tensor(logits), dim=2).cpu().numpy()

    # 예측된 값
    predictions = np.argmax(logits, axis=2)

    # 예측된 태그 디코딩
    pred_tags = [list(tag_id.keys())[p_i] for p in predictions for p_i in p]  # 예측된 인덱스에 해당하는 태그 맵핑
    # 토큰 디코딩
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][0].tolist())  # 토큰화된 숫자(input_ids)를 다시 단어로 변환

    pred_probs = [probs_array[0][i][p] for i, p in enumerate(predictions[0])]  # 각 토큰에 대한 예측 확률 추출

    return tokens, pred_tags, pred_probs
#----------------------------------------------------------------------------------------------------------------------
# def restore_unk(tokens, original_text):
# # UNK 토큰의 위치를 추적하고, 해당 위치에 원본 텍스트를 넣어 복원
#     restored_text = ""
#     token_idx = 0
#     for char in original_text:
#         # 원본 텍스트에서 문자마다 확인하여, UNK 토큰이 나타나면 처리
#         if tokens[token_idx] == '[UNK]':
#             restored_text += char  # UNK를 원본 문자로 복원
#             token_idx += 1
#         else:
#             restored_text += tokens[token_idx]
#             token_idx += 1
#     return restored_text

# # UNK 처리가 될 때도 있고 안될때도 있음 ?? 왜 list out of range가 어느 상황에는 나타나고 어느상황에는 안나타는지 모르겠따
# def deidentify_entities_and_restore_unk(tokens, pred_tags, pred_probs, tokenizer, original_text):
#     deidentified_tokens = []
#     inside_person = False  # Flag to track if we are inside a person name

#     # index_to_ner를 이용하여 정수값을 태그 이름으로 변환
#     pred_tags = [index_to_ner[tag] for tag in pred_tags]  # pred_tags를 정수에서 이름으로 변환

#     for token, tag, prob in zip(tokens, pred_tags, pred_probs):
#         # CLS와 SEP 토큰 건너뜀
#         if token in ['[CLS]', '[SEP]']:
#             continue

#         if token == '_':  # _를 공백으로 변경
#             deidentified_tokens.append(' ')
#             continue

#         # 비식별화 처리 (PER_B, PER_I 태그)
#         if tag in ['PER_B', 'PER_I']:
#             deidentified_tokens.append('*')
#             inside_person = True
#             continue

#         # 서브워드 토큰의 접두사 `##` 제거
#         token = token.replace('##', '')

#         # 일반 토큰 추가
#         deidentified_tokens.append(token)

#     # 최종 복원된 텍스트를 생성
#     deidentified_text = ''.join(deidentified_tokens)
#     deidentified_text = restore_unk(deidentified_tokens, original_text)

#     # 확률을 출력
#     print("\n토큰별 확률:")
#     for token, tag, prob in zip(tokens, pred_tags, pred_probs):
#         print(f"Token: {token}, Tag: {tag}, Probability: {prob:.4f}")

#     return deidentified_text

# if __name__ == "__main__":
#     # 입력된 텍스트 받기
#     text = sys.argv[1]

#     # Step 1: Get tokens, predictions, and probabilities from ner_inference
#     tokens, pred_tags, pred_probs = ner_inference(text)
#     # print('tokens:', tokens)
#     # print('pred_tags:', pred_tags)

#     # Step 2: Apply de-identification and show probabilities (with restore_unk)
#     deidentified_text = deidentify_entities_and_restore_unk(tokens, pred_tags, pred_probs, tokenizer, text)

#     # Print the de-identified text
#     print("\nDe-identified Text:\n", deidentified_text)
#----------------------------------------------------------------------------------------------------------------------
# deidentify_entities_and_restore_unk 함수 코드 수정 필요, token에 대한 tag가 숫자로 나오는 현상 수정(해서 per_b, per_i 태그에 해당하는 글자가 *로 변환되어 출력되도록) (완료)
# 토큰화되어 ##이붙은 단어에서 ##제거 후 깔끔한 문장으로 복원해 비식별조치된 문장 결과 출력되도록 (완료)
# 다시 UNK처리 -> UNK처리하면 이름 식별이 안되고 이름 식별하면 UNK처리가 안되고 무한 반복... 
## kobert 추론 후 처리방식 참고해서 다시 수정하기 UNK에 매핑되는 입력 데이터가 정확히 출력되지 않음

def deidentify_entities_and_restore_unk(tokens, pred_tags, pred_probs, original_text):
    deidentified_text = list(original_text)  # 원본 텍스트를 문자 리스트로 변환
    token_index = 0  # 원본 텍스트 내 위치

    # `pred_tags`를 숫자에서 문자열 태그로 변환
    pred_tags = [index_to_ner[tag] for tag in pred_tags]

    # 디버깅용: 각 토큰별 예측 태그 및 확률 출력
    print("\n[토큰별 예측 결과]")
    print(f"{'Token':<15}{'Tag':<10}{'Probability':<10}")
    print("=" * 35)

    for token, tag, prob in zip(tokens, pred_tags, pred_probs):
        # CLS와 SEP 토큰 건너뛰기
        if token in ['[CLS]', '[SEP]']:
            continue

        # 서브워드 처리: '##' 제거
        if token.startswith('##'):
            token = token[2:]

        # 공백 토큰 처리
        if token == '_':
            continue

        # 예측 결과 출력
        print(f"{token:<15}{tag:<10}{prob:<10.4f}")

        # 이름 태그 처리
        if tag in ['PER_B', 'PER_I']:
            while token_index < len(deidentified_text) and deidentified_text[token_index].isspace():
                token_index += 1  # 공백 건너뛰기

            if token_index < len(deidentified_text):
                # 이름 위치를 '*'로 대체
                deidentified_text[token_index] = '*'
                token_index += 1  # 다음 문자로 이동
        else:
            # 일반 텍스트 처리: 원본 텍스트와 매칭
            while token_index < len(deidentified_text) and deidentified_text[token_index].isspace():
                token_index += 1  # 공백 건너뛰기

            if token_index < len(deidentified_text):
                # [UNK] 처리: 원본 텍스트에서 복원
                if token == '[UNK]':
                    token_index += 1  # [UNK]는 한 문자로 간주
                else:
                    token_index += len(token)  # 토큰 길이만큼 이동

    return ''.join(deidentified_text).strip()


if __name__ == "__main__":
    # `index_to_ner`는 숫자 -> 문자열 태그 매핑
    index_to_ner = {k: v for k, v in tag_id.items()}

    # 입력된 텍스트 받기
    text = sys.argv[1]

    # 예측 결과 가져오기
    tokens, pred_tags, pred_probs = ner_inference(text)

    # 비식별화 수행
    deidentified_text = deidentify_entities_and_restore_unk(tokens, pred_tags, pred_probs, text)

    # 결과 출력
    print("\n비식별 처리된 텍스트:\n", deidentified_text)

## ---------------- UNK 토큰에 해당하는 값 복원까지는 해결, PER_B/PER_I에 해당하는 값만큼의 길이에 대한 *처리가 되도록 수정 필요-----------------
# 위 사항들 수정 완료