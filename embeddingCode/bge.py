from sentence_transformers import SentenceTransformer
import pandas as pd # Make sure pandas is imported
import os # Make sure os is importedrted for normalization
import numpy as np
import torch


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

for id in problem_ids:
    with open(f'/content/drive/My Drive/creative_problem_solving/problem/{id}.txt', 'r', encoding='utf-8') as f:
        problem_text = f.read()
    queries = [get_detailed_instruct(task, problem_text)]
    
    problem_df = df[df['ProblemID'] == id]
    documents = problem_df['Solutions'].tolist()

    input_texts = queries + documents
    
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
    else:
        print("没有检测到 GPU，代码将在 CPU 上运行。")
        
    embeddings = model.encode(input_texts, normalize_embeddings=True, convert_to_tensor=True)
    
    print("Embeddings shape:", embeddings.shape)
    print("前3个归一化后的文本向量：", embeddings[:3])
    
    save_directory = f'/content/drive/My Drive/creative_problem_solving/embeddings/bge/{id}/'
    file_name = 'normalized_text_embeddings.npy' # 使用 .npy 格式
    full_save_path = os.path.join(save_directory, file_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 将 PyTorch 张量转换为 NumPy 数组（如果它还在 GPU 上，需要先移到 CPU）
    embeddings_np = embeddings.cpu().numpy()
    np.save(full_save_path, embeddings_np)
    print(f"归一化后的文本向量已成功保存到: {full_save_path}")