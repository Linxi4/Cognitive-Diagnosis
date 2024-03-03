import torch
import torch.nn as nn
import ast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime
import os
import pandas as pd

resultPath = "../result"
dataPath = "../data/with_code"
now = datetime.datetime.now().strftime('%Y-%m-%d %H %M')
snapPath = resultPath + "/NCDM_C/" + now + "/snapshot"
evalPath = resultPath + "/NCDM_C/" + now
if not os.path.exists(snapPath):
    os.makedirs(snapPath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
S = 3824
# E = 5774
E = 5977
K = 28
# codeBert的输出维度
D = 768
# 代码所占维度
d1 = 256
d2 = 128
d3 = K

lr = 0.01
epoch = 10
batch_size = 512
shuffle = True

with open(evalPath + '/param.txt', 'a', encoding='utf8') as f:
    f.write('v2\n')
    f.write('learn_rate= %f, epoch= %d, batch_size= %d, shuffle= %s \n' % (lr, epoch, batch_size, shuffle))

# 加载题目代码的embedding
with open('../data/with_code/code_embeddings.json', 'r') as f:
    code_embeddings_dict = json.load(f)
with open('../data/with_code/text_embeddings.json', 'r') as f:
    text_embeddings_dict = json.load(f)


class Net(nn.Module):
    """
    NeuralCDM
    """

    def __init__(self):
        self.len1, self.len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(S, K)
        # self.exer_emb = nn.Embedding(E, K)


        self.code_emb1 = nn.Linear(D, d1)
        self.code_emb2 = nn.Linear(d1, d2)
        self.code_emb3 = nn.Linear(d2, d3)
        self.text_emb1 = nn.Linear(D, d1)
        self.text_emb2 = nn.Linear(d1, d2)
        self.text_emb3 = nn.Linear(d2, d3)
        self.exer_emb = nn.Linear(d3*2, K)
        self.e_discrimination = nn.Linear(K, 1)

        self.prednet_full1 = nn.Linear(K, self.len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.len1, self.len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, code_embedding, text_embedding):
        """
        :param code_embedding: 参考代码embedding
        :param text_embedding: 题目文本embedding
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        """
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        # exer_emb = torch.sigmoid(self.exer_emb(exer_id))

        code_emb = torch.sigmoid(self.code_emb3(self.code_emb2(self.code_emb1(code_embedding))))
        text_emb = torch.sigmoid(self.text_emb3(self.text_emb2(self.text_emb1(text_embedding))))
        exer_emb = self.exer_emb(torch.cat((code_emb,text_emb),dim=1))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_emb)) * 10

        # 交互函数
        r = e_discrimination * (stu_emb - exer_emb * kn_emb)


        # prednet
        # input_x = e_discrimination * (stu_emb - torch.cat((k_difficulty, code_emb), dim=1)) * kn_emb
        # input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        input_x = (torch.sigmoid(self.prednet_full1(r)))
        input_x = (torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class MyDataSet(Dataset):
    def __init__(self, file):
        with open(file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        knowledge_emb = [0.] * K
        log = self.data[idx]
        for knowledge_code in log['knowledge_code']:
            knowledge_emb[knowledge_code - 1] = 1.0
        code_embedding = code_embeddings_dict[str(log["exer_id"])]
        text_embedding = text_embeddings_dict[str(log["exer_id"])]
        return torch.tensor(log['user_id'] - 1).to(device), torch.tensor(log["exer_id"] - 1).to(device), torch.tensor(
            knowledge_emb).to(device), torch.tensor(code_embedding).to(device), torch.tensor(text_embedding).to(
            device), torch.tensor(int(log['score'])).to(device)


def train():
    train_set = MyDataSet(dataPath + "/train_set.json")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    print('training model...')

    loss_function = nn.NLLLoss()
    for i in range(epoch):
        batch_cnt = 0
        running_loss = 0.0
        net.train()
        for batch in train_loader:
            optimizer.zero_grad()

            user_ids, exer_ids, kn_embs, code_embeddings, text_embeddings, labels = batch

            output_1 = net.forward(user_ids, exer_ids, kn_embs, code_embeddings, text_embeddings)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            # grad_penalty = 0
            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()

            net.apply_clipper()
            running_loss += loss.item()
            batch_cnt += 1

            if batch_cnt % 200 == 0:
                print('[%d, %5d] loss: %.3f' % (i + 1, batch_cnt, running_loss / 200))
                running_loss = 0.0

        validate(net, i + 1)
        save_snapshot(net, snapPath + "/epoch_" + str(i + 1))


def validate(net, ep):
    val_set = MyDataSet(dataPath + "/val_set.json")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    for batch in val_loader:
        user_ids, exer_ids, kn_embs, code_embeddings, text_embeddings, labels = batch

        output = net.forward(user_ids, exer_ids, kn_embs, code_embeddings, text_embeddings)

        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (ep, accuracy, rmse, auc))
    with open(evalPath + '/val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (ep, accuracy, rmse, auc))


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    train()
