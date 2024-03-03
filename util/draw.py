import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from matplotlib.font_manager import FontProperties
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']


# 读取JSON文件
with open('../data/with_code/log_data.json', 'r') as f:
    data = json.load(f)

for log in data:
    user_id = log['user_id']
    if(user_id == 723):
        print(log)
