# coding: utf-8
# 2021/3/23 @ tongshiwei
import json
import logging
from EduCDM import GDDINA
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

batch_size = 32
S = 3824 + 1
# E = 5774
E = 5977 + 1
K = 28

def code2vector(x):
    vector = [0] * K
    for k in x:
        vector[k - 1] = 1
    return vector





def transform(x, y, z, k, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train_data = json.load(open("../data/with_code/train_set.json", encoding='utf8'))
valid_data = json.load(open("../data/with_code/val_set.json", encoding='utf8'))
test_data = json.load(open("../data/with_code/test_set.json", encoding='utf8'))

train_data = pd.DataFrame(train_data)
valid_data = pd.DataFrame(valid_data)
test_data = pd.DataFrame(test_data)
train_data['knowledge_code'] = train_data['knowledge_code'].apply(lambda x: code2vector(x))
valid_data['knowledge_code'] = valid_data['knowledge_code'].apply(lambda x: code2vector(x))
test_data['knowledge_code'] = test_data['knowledge_code'].apply(lambda x: code2vector(x))

train, valid, test = [
    transform(data["user_id"], data["exer_id"], data["score"], data["knowledge_code"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = GDDINA(S, E, K)

cdm.train(train, valid, epoch=2)
cdm.save("dina.params")

cdm.load("dina.params")
accuracy, rmse, mae, auc = cdm.eval(test)
print("acc, RMSE, MAE, auc are %.6f, %.6f, %.6f, %.6f" % (accuracy, rmse, mae, auc))
