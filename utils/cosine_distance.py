import pandas as pd
import numpy as np

def get_embeddings(model_name, problem_id, label = None):
    if label is None:
        full_save_path = f"Data/embeddings/{model_name}/{problem_id}/normalized_text_embeddings.npy"
    else:
        full_save_path = f"Data/embeddings/{label}/{model_name}/{problem_id}/normalized_text_embeddings.npy"
    loaded_embeddings = np.load(full_save_path)
    
    query_embedding = loaded_embeddings[0]
    solutions_embeddings = loaded_embeddings[1:]
    return query_embedding, solutions_embeddings

def get_prompt_distance(model_name, problem_id, label = None):
    query_embedding, solutions_embeddings = get_embeddings(model_name, problem_id, label)
    distances = 1 - solutions_embeddings @ query_embedding
    return distances

def get_peer_distance_from_embeddings(embeddings):
    sum_of_dot_products = embeddings @ embeddings.sum(0) - (embeddings ** 2).sum(1)
    peer_cos_distances = 1 - sum_of_dot_products / (embeddings.shape[0] - 1)
    return peer_cos_distances

def get_peer_distance(model_name, problem_id, label = None):
    query_embedding, solutions_embeddings = get_embeddings(model_name, problem_id, label)
    peer_distances = get_peer_distance_from_embeddings(solutions_embeddings)
    return peer_distances.tolist()