import os
import torch
import numpy as np
import torch.nn.functional as F

from vllm import LLM
from typing import List


def embed_solutions(problem_ID: str, solutions: List[str], save_path: str) -> None:
    # Check if the provided problem ID exists
    all_problem_IDs = os.listdir("Data/problem")
    assert problem_ID + ".txt" in all_problem_IDs, f"{problem_ID} does not exist."
    
    # Load the model, and embed the solutions
    model = LLM(model="Qwen/Qwen3-Embedding-0.6B", task="embed")
    outputs = model.embed(solutions)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    # Normalize, turn into numpy format, and save the embeddings
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    numpy_embeddings = normalized_embeddings.cpu().numpy()
    np.save(save_path, numpy_embeddings)
