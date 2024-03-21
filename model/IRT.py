# coding: utf-8
# 2021/5/2 @ liujiayu
import logging
import json
import numpy as np
import pandas as pd
from EduCDM import EMIRT

S = 3824
E = 5977

train_data = json.load(open("../data/with_code/train_set.json", encoding='utf8'))
valid_data = json.load(open("../data/with_code/val_set.json", encoding='utf8'))
test_data = json.load(open("../data/with_code/test_set.json", encoding='utf8'))

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
R = -1 * np.ones(shape=(S, E))
R[train_data['user_id'] - 1, train_data['exer_id'] - 1] = train_data['score']

test_set = []
for i in range(len(test_data)):
    row = test_data.iloc[i]
    test_set.append({'user_id': int(row['user_id']) - 1, 'item_id': int(row['exer_id']) - 1, 'score': row['score']})

logging.getLogger().setLevel(logging.INFO)

cdm = EMIRT(R, S, E, dim=1, skip_value=-1)  # IRT, dim > 1 is MIRT

cdm.train(lr=1e-3, epoch=2)
cdm.save("irt.params")

cdm.load("irt.params")
accuracy, rmse, mae, auc = cdm.eval(test_set)
print("acc, RMSE, MAE, auc are %.6f, %.6f, %.6f, %.6f" % (accuracy, rmse, mae, auc))

# # ---incremental training
# new_data = [{'user_id': 0, 'exer_id': 2, 'score': 0.0}, {'user_id': 1, 'exer_id': 1, 'score': 1.0}]
# cdm.inc_train(new_data, lr=1e-3, epoch=2)

# ---evaluate user's state
stu_rec = np.random.randint(-1, 2, size=E)
dia_state = cdm.transform(stu_rec)
print("user's state is " + str(dia_state))
