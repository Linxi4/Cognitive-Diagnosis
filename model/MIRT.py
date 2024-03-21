# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from EduCDM import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import json

train_data = json.load(open("../data/with_code/train_set.json", encoding='utf8'))
valid_data = json.load(open("../data/with_code/val_set.json", encoding='utf8'))
test_data = json.load(open("../data/with_code/test_set.json", encoding='utf8'))

train_data = pd.DataFrame(train_data)
valid_data = pd.DataFrame(valid_data)
test_data = pd.DataFrame(test_data)

batch_size = 256
S = 3824 + 1
E = 5977 + 1
K = 28

def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["exer_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = MIRT(S, E, K)

cdm.train(train, valid, epoch=2)
cdm.save("mirt.params")

cdm.load("mirt.params")
accuracy, rmse, mae, auc = cdm.eval(test)
print("acc, RMSE, MAE, auc are %.6f, %.6f, %.6f, %.6f" % (accuracy, rmse, mae, auc))
