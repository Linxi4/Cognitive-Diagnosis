import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import json

# 检查CUDA是否可用，并据此设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器，使用CodeBERT的模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base').to(device)
model.eval()  # 设置为评估模式

# 读取CSV文件
df = pd.read_csv('../data/origin/题目数据集.csv')

# 过滤掉代码为空的记录
df = df.dropna(subset=['标准答案'])

# 确保代码列是字符串类型
df['标准答案'] = df['标准答案'].astype(str)

# 创建字典来存储结果
code_embeddings_dict = {}

# 设置批处理大小
batch_size = 16

# 准备数据批次
for start_index in range(0, len(df), batch_size):
    end_index = start_index + batch_size
    batch = df.iloc[start_index:end_index]

    # 批量处理代码
    inputs = tokenizer(batch['标准答案'].tolist(), return_tensors='pt', padding=True, truncation=True,
                       max_length=512).to(device)

    # 使用模型获取向量表示
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取[CLS]标记的向量作为代码片段的表示，并移动到CPU上
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 更新字典
    for idx, row in enumerate(batch.itertuples()):
        code_embeddings_dict[int(row.题目id)] = embeddings[idx].tolist()

# 将字典保存到JSON文件中
with open('../data/with_code/code_embeddings.json', 'w') as f:
    json.dump(code_embeddings_dict, f)