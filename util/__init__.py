import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载 CodeBERT 模型和 tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)


# 自定义数据集类
class CodeDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        code = self.dataframe.iloc[idx]['code']

        return {
            'exer_id': self.dataframe.iloc[idx]['exer_id'],
            'code': code
        }


# 读取题目数据集
df = pd.read_csv("../data/origin/题目数据集.csv", encoding='utf8')
df = df[['题目id', '标准答案']]
df = df.rename(columns={'题目id': 'exer_id', '标准答案': 'code'})
df['label'] = df['code'].notna()
df = df[df['code'].notna()]
# 创建数据集实例
code_dataset = CodeDataset(df)

# 创建 DataLoader 实例
batch_size = 8
dataloader = DataLoader(code_dataset, batch_size=batch_size, shuffle=False)

# 初始化一个列表来保存结果
result_list = []

# 使用 DataLoader 进行批处理
with torch.no_grad():
    model.eval()
    for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        exer_id = batch['exer_id']
        code = batch['code']

        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(device)

        # 获取模型输出
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # 如果你想使用整个代码片段的向量表示，你可以对最后一层的隐藏状态进行平均或池化
        code_embedding = torch.mean(last_hidden_states, dim=1)
        for i in range(len(batch['exer_id'])):
            result_list.append({
                'exer_id': batch['exer_id'][i].item(),
                'embedding': code_embedding[i].tolist()
            })

# 将结果存储到 CSV 文件
result_df = pd.DataFrame(result_list)
result_df.to_csv("../data/with_code/code_embedding.csv", index=False)
