import json
code_embedding_path = '../data/with_code/code_embeddings.json'
text_embedding_path = '../data/with_code/text_embeddings.json'
# 读取JSON文件并将内容保存到字典中
with open(code_embedding_path, 'r') as f:
    code_embeddings_dict = json.load(f)
print(len(code_embeddings_dict))
with open(text_embedding_path, 'r') as f:
    text_embeddings_dict = json.load(f)
    print(len(text_embeddings_dict))