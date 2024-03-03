import math
import random
import json
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader

min_log = 10
max_log = 1842

device = torch.device("cpu")


def divide_data():
    """
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    """
    with open('../data/filtered_no_code/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        stu_i += 1
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, val_slice, test_slice, train_set, val_set, test_set = [], [], [], [], [], []
    max_exer_id = 0
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_val = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.7)
        val_size = int(stu['log_num'] * 0.1)
        test_size = stu['log_num'] - train_size - val_size
        logs = []
        for log in stu['logs']:
            max_exer_id = max(max_exer_id, log['exer_id'])
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_val['log_num'] = val_size
        stu_val['logs'] = logs[train_size:train_size + val_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[-test_size:]
        train_slice.append(stu_train)
        val_slice.append(stu_val)
        test_slice.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            max_exer_id = max(max_exer_id, log['exer_id'])
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        for log in stu_val['logs']:
            max_exer_id = max(max_exer_id, log['exer_id'])
            val_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                            'knowledge_code': log['knowledge_code']})
        for log in stu_test['logs']:
            max_exer_id = max(max_exer_id, log['exer_id'])
            test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                             'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    print(max_exer_id)
    # with open('../data/slice/train_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    # with open('../data/slice/val_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(val_slice, output_file, indent=4, ensure_ascii=False)
    # with open('../data/slice/test_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(test_slice, output_file, indent=4, ensure_ascii=False)
    with open('../data/filtered_no_code/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/filtered_no_code/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)  # 直接用test_set作为val_set
    with open('../data/filtered_no_code/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


def showData():
    with open('../data/log_data.json', encoding='utf8') as i_f:
        logs = json.load(i_f)
    min_log_num = 1000
    max_log_num = 0
    for log in logs:
        max_log_num = max(max_log_num, log['log_num'])
        min_log_num = min(min_log_num, log['log_num'])
    print(min_log_num)
    print(max_log_num)


def filterData():
    # 过滤掉没有参考答案
    df = pd.read_csv("../data/origin/题目数据集.csv", encoding='utf8')
    df = df[df['标准答案'].notna()]
    question_dict = set(df['题目id'])
    with open('../data/log_data.json', 'r') as json_file:
        log_data = json.load(json_file)
    for index, value in enumerate(log_data):
        logs = value['logs']

        new_log_num = 0
        new_logs = []

        for log in logs:
            exer_id = log['exer_id']
            if exer_id in question_dict:
                new_log_num += 1
                new_logs.append(log)

        log_data[index]['logs'] = new_logs
        log_data[index]['log_num'] = new_log_num
    with open('../data/filtered_no_code/log_data.json', 'w') as json_file:
        json.dump(log_data, json_file, indent=2)
    pass


def tryBert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)

    nl_tokens = tokenizer.tokenize("return maximum value")
    code_tokens = tokenizer.tokenize("using namespace std;""int main(){""int j[102425]={0},s[102425]={0},i,l,n,x,y,"
                                     "w[102425]={0};""	cin>>n;""	for(i=1;i<=3;i++){""		cin>>j[i]>>s[i];""	"
                                     "}""	for(i=1;i<=3;i++){""		if(n%j[i]==0)""		w[i]=n/j[i]*s[i];""		"
                                     "else""		w[i]=(n/j[i]+1)*s[i];""	}""    for(i=1;i<3;i++){""		for("
                                     "l=i+1;l<=3;l++){""			if(w[i]>w[l]){""				x=w[l];""		"
                                     "		w[l]=w[i];""				w[i]=x;""    }""}""	}""	cout<<w[1];""}")
    tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None, :].to(device))[0]
    print(context_embeddings)


class SubmitRecord(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file, encoding='utf-8')
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        log = self.data[idx]
        return torch.tensor(log['user_id'] - 1).to(device), torch.tensor(log["exer_id"] - 1).to(device) \
            , torch.tensor(int(log['score'])).to(device)


def codeBertEnCoder(code):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)

    code_tokens = tokenizer.tokenize(code)
    tokens_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    embedding = model(torch.tensor(tokens_ids)[None, :].to(device))[0]
    return embedding


if __name__ == '__main__':
    # filterData()
    # divide_data()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

