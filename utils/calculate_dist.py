import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def Jaccard(A, B):
    set_A = set(str(A).split())
    set_B = set(str(B).split())
    intersection = set_A & set_B
    union = set_A | set_B
    if not union:
        return 0.0
    return len(intersection) / len(union)


def get_embeddings(ID):
    # model: "Qwen3-Embedding-0.6B"
    full_save_path = f"Data/embeddings/{ID}/normalized_text_embeddings.npy"
    loaded_embeddings = np.load(full_save_path)
    
    query_embedding = loaded_embeddings[0]
    solutions_embeddings = loaded_embeddings[1:]
    return query_embedding, solutions_embeddings


def get_semantic_similarity(ID, mode="DSI"):
    if mode == "cosine_distance":
        query_embedding, solutions_embeddings = get_embeddings(ID)
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            solutions_embeddings
        )[0]

        return similarities.tolist()
    
    elif mode == "DSI":
        df = pd.read_csv('Data/CPSTfulldataset2.csv')
        dsi_values = df[df['ProblemID'] == ID]['DSI'].tolist()
        return dsi_values


def get_peer_distance(ID, solutions_list, mode, number_of_peers=0):
    if number_of_peers == 0:
        number_of_peers = len(solutions_list)
        
    if mode == "cosine_distance":
        query_embedding, solutions_embeddings = get_embeddings(ID)
        avg_distances = []

        for i, emb_i in enumerate(solutions_embeddings):
            distances = []
            counter = 0
            for j, emb_j in enumerate(solutions_embeddings):
                if i != j:
                    dist = 1 - cosine_similarity(
                        emb_i.reshape(1, -1),
                        emb_j.reshape(1, -1)
                    )[0][0]
                    distances.append(dist)
                    counter += 1
                    if counter >= number_of_peers:
                        break
            avg_distance = np.mean(distances) if distances else 0.0
            avg_distances.append(avg_distance)
        return avg_distances
    
    elif mode == "Jaccard":
        avg_jaccard_scores = []
        for i, sol_i in enumerate(solutions_list):
            scores = []
            counter = 0
            for j, sol_j in enumerate(solutions_list):
                if i != j:
                    score = Jaccard(sol_i, sol_j)
                    scores.append(score)
                    counter += 1
                    if counter >= number_of_peers:
                        break
            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_jaccard_scores.append(avg_score)
        return avg_jaccard_scores
