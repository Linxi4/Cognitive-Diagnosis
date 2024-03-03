import math
import random
from tqdm import tqdm
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

exer_path = '../data/origin/题目数据集.csv'
log_path = '../data/origin/学生序列数据.txt'
record_path = '../data/origin/学员提交记录数据_去掉0号题.csv'
exer2ccpt_path = '../data/origin/exer_id2ccpt_id.csv'

folder = '../data/with_code'


def get_uid_num():
    # 获取user_id与用户编号之间的映射关系
    uid2num = {}
    num2uid = {}
    num = 1
    with open(log_path, 'r', ) as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='获取用户id与用户编号的映射字典'):
            uid = line.split(',')[0]
            uid2num[uid] = num
            num2uid[num] = uid
            num += 1
    return uid2num, num2uid


def get_kn():
    # 获取题目与知识点之间的关系
    exer2kn = {}
    with open(exer2ccpt_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='获取题目的知识点列表'):
            items = line.split(',')
            exer_id = int(items[0])
            kn = [int(item) for item in items[1:]]
            exer2kn[exer_id] = kn
    return exer2kn


def generate_records(uid2num, exer2kn):
    """
    1. 将用户id转为用户编号
    2. 存入题目对应知识点概念列表
    3. 过滤掉不需要的列信息
     :return: records:exer_id,user_id,result,score,code,knowledge_code
    """
    # 处理学员提交记录数据_去掉0号题.csv文件，保留用户代码，将用户id转为用户编号，保存题目对应的知识点列表
    with open(record_path, 'r', encoding='utf8') as f:
        df = pd.read_csv(f)
        df = df[['问题id', '学员id', '判题结果', '用例正确率', '源代码']]
        df = df.rename(
            columns={'问题id': 'exer_id', '学员id': 'user_id', '判题结果': 'result', '用例正确率': 'score', '源代码': 'code'})
        df['user_id'] = df['user_id'].apply(lambda x: uid2num[x])
        df['knowledge_code'] = df['exer_id'].apply(lambda x: exer2kn.get(x, []))
        records = df.to_dict(orient='records')
    return records


def create_records():
    uid2num, num2uid = get_uid_num()
    exer2kn = get_kn()
    records = generate_records(uid2num, exer2kn)
    print('写入文件')
    with open(folder + '/uid2num.json', 'w', encoding='utf8') as json_file:
        json.dump(uid2num, json_file, indent=2)
    with open(folder + '/num2uid.json', 'w') as json_file:
        json.dump(num2uid, json_file, indent=2)
    with open(folder + '/exer2kn.json', 'w') as json_file:
        json.dump(exer2kn, json_file, indent=2)
    with open(folder + '/records.json', 'w', encoding='utf8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=2)


def show_data():
    exer_info = pd.read_csv(open(exer_path, 'r', encoding='utf8'))
    exer_dict = exer_info.set_index('题目id')['标准答案'].to_dict()
    logs = json.load(open(folder + '/log_data_code.json', 'r', encoding='utf8'))
    size = len(logs)
    user_dict = {}  # 记录用户的做题数量
    exer_dict_kn = {}  # 记录没有知识点的题目
    exer_dict_code = {}  # 记录没有参考答案的题目
    for log in tqdm(logs):
        exer_id = log.get('exer_id')
        user_id = log.get('user_id')
        kn = log.get('knowledge_code')
        user_dict[user_id] = user_dict.get(user_id, 0) + 1
        if len(kn) == 0:
            exer_dict_kn[exer_id] = exer_dict_kn.get(exer_id, 0) + 1
        if len(exer_dict.get(exer_id, "")) < 10:
            exer_dict_code[exer_id] = exer_dict_code.get(exer_id, 0) + 1
    pass


def clean_data():
    """
    1. 题目未关联知识点的记录
    2. 题目没有参考代码的记录
    :return:
    """
    exer_info = pd.read_csv(open(exer_path, 'r', encoding='utf8'))
    # 过滤出题目描述为空的行
    empty_description_df = exer_info[exer_info['题目描述'].isna()]
    empty_code_df = exer_info[exer_info['标准答案'].isna()]
    # 获取这些行的题目id，并存入集合中
    to_remove = set(empty_description_df['题目id'])
    to_remove = to_remove.union(set(empty_code_df['题目id']))

    cnt = 0
    records = json.load(open(folder + '/records.json', 'r', encoding='utf8'))
    new_records = []
    # 1. 删除未关联知识点的记录
    for record in tqdm(records):
        exer_id = record.get('exer_id')
        kn = record.get('knowledge_code')
        if len(kn) > 0 and exer_id not in to_remove:
            new_records.append(record)
            cnt += 1
    print(cnt)
    with open(folder + '/records_filter.json', 'w', encoding='utf8') as json_file:
        json.dump(new_records, json_file, ensure_ascii=False, indent=2)
    pass


def format_data():
    """
    将做题记录按用户划分
    过滤掉做题记录数量小于10的
    :return:
    """
    records = json.load(open(folder + '/records_filter.json', 'r', encoding='utf8'))
    log_data = {}
    for record in tqdm(records):
        user_id = record.get('user_id')
        if user_id not in log_data.keys():
            log_data[user_id] = {"user_id": user_id, "log_num": 0, "logs": []}
        log_data[user_id]["log_num"] += 1
        log_data[user_id]["logs"].append(record)
    log_data = list(log_data.values())
    print(len(log_data))
    logs = []
    for log in log_data:
        log_num = log.get("log_num")
        if log_num >= 10:
            logs.append(log)
    print(len(logs))
    with open(folder + '/log_data.json', 'w', encoding='utf8') as json_file:
        json.dump(logs, json_file, ensure_ascii=False, indent=2)


def divide_data():
    """
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    """
    with open('../data/with_code/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, val_slice, test_slice, train_set, val_set, test_set = [], [], [], [], [], []
    max_exer_id = 0
    max_user_id = 0
    for stu in tqdm(stus):
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
            max_user_id = max(max_user_id, log['user_id'])
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
    print(max_user_id)
    # with open('../data/slice/train_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    # with open('../data/slice/val_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(val_slice, output_file, indent=4, ensure_ascii=False)
    # with open('../data/slice/test_slice.json', 'w', encoding='utf8') as output_file:
    #     json.dump(test_slice, output_file, indent=4, ensure_ascii=False)
    with open('../data/with_code/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=2, ensure_ascii=False)
    with open('../data/with_code/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=2, ensure_ascii=False)  # 直接用test_set作为val_set
    with open('../data/with_code/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=2, ensure_ascii=False)


def te():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

    data = json.load(open('../data/with_code/log_data.json', encoding='utf8'))
    code_dict = {}
    new_data = []
    submit_id = 1
    for user in tqdm(data):
        user_id, log_num, logs = user['user_id'], user['log_num'], user['logs']
        temp = {'user_id': user_id, 'log_num': log_num}
        new_logs = []
        for log in logs:
            log['submit_id'] = submit_id
            log['score'] = 1 if log['score'] == 1.0 else 0
            exer_id, user_id, score, kn = log['exer_id'], log['user_id'], log['score'], log['knowledge_code']
            code_dict[submit_id] = log['code']
            new_log = {'exer_id': exer_id, 'user_id': user_id, 'score': score, 'submit_id': submit_id,
                       'knowledge_code': kn}
            new_logs.append(new_log)
            submit_id += 1
        temp['logs'] = new_logs
        new_data.append(temp)
    with open('../data/with_code/new_log_data.json', 'w', encoding='utf8') as output_file:
        json.dump(new_data, output_file, indent=2, ensure_ascii=False)

    data = []
    submit_ids, codes = zip(*code_dict.items())
    batch_size = 128
    for idx in tqdm(range(0, len(submit_ids), batch_size)):
        # 处理批量代码片段
        ids = submit_ids[idx:idx + batch_size]
        code = codes[idx:idx + batch_size]
        token = tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(device)

        # 提取表示
        with torch.no_grad():
            output = model(**token)

        embedding = output.last_hidden_state[:, 0, :]  # 获取批量CLS向量
        for i in range(len(ids)):
            data.append({"submit_id": ids[i],
                         "code": code[i],
                         "embedding": embedding[i]})
    df_embedding = pd.DataFrame(data)
    df_embedding.to_csv(open('../data/with_code/submit_embedding.csv', 'w', encoding='utf8'))


if __name__ == '__main__':
    # create_records()
    # show_data()
    clean_data()
    format_data()
    divide_data()
    # te()
