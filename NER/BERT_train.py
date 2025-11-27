# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer, AutoModelForTokenClassification, EarlyStoppingCallback
import pandas as pd
import numpy as np
import os

#cuda 사용 가능여부 확인
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
# %%
tag_id = {tag : i for tag, i in enumerate(label)}
id_tag = {i : tag for tag, i in tag_id.items()}

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
# %%
train_dataset = torch.load('./data/BERT_train_dataset.pth')
test_dataset = torch.load('./data/BERT_test_dataset.pth')

training_args = TrainingArguments(
    output_dir='./Bert_base_4_25.01.06',
    num_train_epochs=5,
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
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = len(tag_id))
model.to(device)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

model_dir = os.path.join(os.getcwd(), "models/BERT_4")

os.makedirs(model_dir, exist_ok=True)

trainer.save_model(model_dir)
print(f"Model saved at : {model_dir}")