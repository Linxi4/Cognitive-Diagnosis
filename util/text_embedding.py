import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import json

# 检查CUDA是否可用，并据此设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器，将模型转移到指定的设备
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
model = BertModel.from_pretrained('hfl/chinese-bert-wwm').to(device)
model.eval()  # 设置为评估模式

# 读取CSV文件
df = pd.read_csv('../data/origin/题目数据集.csv')
df = df.dropna(subset=['题目描述'])
# 创建字典来存储结果
embeddings_dict = {}

# 设置批处理大小
batch_size = 16

# 准备数据批次
for start_index in tqdm(range(0, len(df), batch_size)):
    # 计算结束索引
    end_index = start_index + batch_size
    batch = df.iloc[start_index:end_index]

    # 批量处理文本
    inputs = tokenizer(batch['题目描述'].tolist(), return_tensors='pt', padding=True, truncation=True,
                       max_length=512).to(device)

    # 使用模型获取向量表示
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取[CLS]标记的向量作为句子表示，并移动到CPU上
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 更新字典
    for idx, row in enumerate(batch.itertuples()):
        embeddings_dict[int(row.题目id)] = embeddings[idx].tolist()

# 将字典保存到JSON文件中
with open('../data/with_code/text_embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f)
