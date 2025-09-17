import os
import torch
import numpy as np
import torch.nn.functional as F

from vllm import LLM
from typing import List
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer


def get_bert_embeddings_batched(texts, batch_size=32):
    """
    Encodes a list of texts into embeddings using a BERT model with mean pooling,
    处理方式为分批次（batch）进行，以优化内存。
    """
    all_embeddings = []
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained("bert-large-uncased")

    # 将文本分成小批次
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize the current batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

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


def embed_solutions(problem_ID: str, solutions: List[str], embedding_model_label: str, embedding_model: str, save_path: str) -> None:
    # Currently only bert/bge/Qwen are supported as embedding models.
    assert embedding_model_label in ["bert", "bge", "Qwen"], f"Unsupported embedding model: {embedding_model_label}"
    
    # Check if the provided problem ID exists
    all_problem_IDs = os.listdir("Data/problem")
    assert problem_ID + ".txt" in all_problem_IDs, f"{problem_ID} does not exist."
    
    # Load the model, and embed the solutions
    if embedding_model_label == "bert":
        embeddings = get_bert_embeddings_batched(solutions, batch_size=32)
    elif embedding_model_label == "bge":
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(solutions, normalize_embeddings=True, convert_to_tensor=True)
    elif embedding_model_label == "Qwen":
        model = LLM(model=embedding_model, task="embed")
        outputs = model.embed(solutions)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    # Normalize, turn into numpy format, and save the embeddings
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    numpy_embeddings = normalized_embeddings.cpu().numpy()
    np.save(save_path, numpy_embeddings)
