# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, EarlyStoppingCallback
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #dataset 불러올 때 경고창 무시하려고 임시

if torch.cuda.is_available():    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# %%
train_df = pd.read_csv('./data/train_data.csv', index_col=0)
label = []

def find_label(labels):
    for i in labels:
        if i not in label:
            label.append(i)

train_df['tag'].str.split().apply(find_label)
label.sort()
label = label + ['POS'] + ['ADJ'] + ['SUF']

tag_id = {tag : i for tag, i in enumerate(label)}
id_tag = {i : tag for tag, i in tag_id.items()}
# print('tag_id :', tag_id)
# print('id_tag :', id_tag)

# %%
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key : torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = torch.load('./data/BERT_train_dataset.pth')
test_dataset = torch.load('./data/BERT_test_dataset.pth')
# print("Train dataset labels:", np.unique(train_dataset.labels))
# print("Test dataset labels:", np.unique(test_dataset.labels))

training_args = TrainingArguments(
    output_dir='./result2',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_steps=1000,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=5,
    save_strategy='steps',
    eval_strategy='steps',
    save_steps=1000,
    eval_steps=1000, #1000스텝마다 평가 수행 시 로깅 스텝에 맞춰 결과가 출력됨
    load_best_model_at_end=True
)

# %%
model_dir = './models/BERT_4'
model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=len(tag_id))
model.to(device)

# print("모델 출력 클래스 수(num_labels):", model.config.num_labels)
# print("Tag ID 크기:", len(tag_id))

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# 평가 수행
evaluation_results = trainer.evaluate()
print("평가 결과 :\n", evaluation_results)

#예측 수행
predictions = trainer.predict(test_dataset)
# print(predictions.predictions.shape, predictions.label_ids.shape)

pred = np.argmax(predictions.predictions, axis=-1)

# %% 최종 출력본 (UNK 포함되어 있어서 warning 뜸)
# index_to_ner: 숫자 키 -> 문자열 값 매핑
index_to_ner = {i: tag for tag, i in tag_id.items()}

# val_tags_l과 y_predicted_l을 숫자 형태로 유지
val_tags_l = np.ravel(predictions.label_ids).astype(int).tolist()
y_predicted_l = np.ravel(pred).astype(int).tolist()

precision_weighted = precision_score(val_tags_l, y_predicted_l, average='weighted', labels=[22, 23])
recall_weighted = recall_score(val_tags_l, y_predicted_l, average='weighted', labels=[22, 23])
f1_weighted = f1_score(val_tags_l, y_predicted_l, average='weighted', labels=[22, 23])

# labels와 target_names 생성
labels = list(tag_id.values())  # 숫자 레이블 [0, 1, 2, ...]
target_names = [index_to_ner[label] for label in labels]  # 문자열 라벨 ['UNK', 'AFW_B', ...]

# Debugging
print("Labels (문자열 라벨):", labels)
print("Target Names (숫자 레이블):", target_names)

# classification_report 호출
print(classification_report(val_tags_l, y_predicted_l, labels=target_names, 
                            target_names=labels, zero_division=0))
print('정확도 :', accuracy_score(val_tags_l, y_predicted_l))
print('precision :', round(precision_weighted, 4))
print('recall :', round(recall_weighted, 4))
print('f1 score :', round(f1_weighted,4))