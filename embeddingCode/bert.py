import pandas as pd
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a problem scenario, generate a single solution to solve it'

# No need to add instruction for retrieval documents
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/creative_problem_solving/CPSTfulldataset2.csv'

df = pd.read_csv(file_path)
problem_ids = df["ProblemID"].unique()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")

# Check for GPU and move model to device if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("没有检测到 GPU，代码将在 CPU 上运行。")

def get_bert_embeddings_batched(texts, batch_size=32):
    """
    Encodes a list of texts into embeddings using a BERT model with mean pooling,
    处理方式为分批次（batch）进行，以优化内存。
    """
    all_embeddings = []

    # 将文本分成小批次
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the current batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)

        # Get model outputs
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform mean pooling on the token embeddings
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask

        # Normalize and append to the list
        all_embeddings.append(F.normalize(batch_embeddings, p=2, dim=1))

    # Concatenate all batch embeddings into a single tensor
    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        return torch.tensor([])

for id in problem_ids:
    with open(f'/content/drive/My Drive/creative_problem_solving/problem/{id}.txt', 'r', encoding='utf-8') as f:
        problem_text = f.read()
    queries = [get_detailed_instruct(task, problem_text)]

    problem_df = df[df['ProblemID'] == id]
    documents = problem_df['Solutions'].tolist()

    input_texts = queries + documents

    embeddings = get_bert_embeddings_batched(input_texts)

    print("Embeddings shape:", embeddings.shape)
    print("前3个归一化后的文本向量：", embeddings[:3])

    save_directory = f'/content/drive/My Drive/creative_problem_solving/embeddings/bert/{id}/'
    file_name = 'normalized_text_embeddings.npy' # 使用 .npy 格式
    full_save_path = os.path.join(save_directory, file_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert PyTorch tensor to NumPy array
    embeddings_np = embeddings.cpu().numpy()
    np.save(full_save_path, embeddings_np)
    print(f"归一化后的文本向量已成功保存到: {full_save_path}")