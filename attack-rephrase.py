import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from string import Template

from utils.query import test
from utils.dsi import get_dsi
from utils.embed import embed_solutions
from query_originality_zero_shot import getOutputs
from compare_performance import get_data, calculate_performance, draw, show_statistic
from utils.cosine_distance import get_prompt_distance_from_embeddings, get_peer_distance_from_embeddings


# Specify the LLM model to be leveraged for zero-shot originality evaluation test.
model = "google/gemini-2.5-flash"

# Step 1: Generate attacked solutions (instructions to require high scores injected)
# and compute DSI for each attacked solution.
if os.path.exists("Data/CPSTfulldataset2_rephrase_attacked.csv"):
    original_dataset = pd.read_csv("Data/CPSTfulldataset2_rephrase_attacked.csv")
else:
    original_dataset = pd.read_csv("Data/CPSTfulldataset2.csv")
    
    # In the very first beginning, complement the original prompt & peer cosine distances.
    final = pd.read_csv("Data/final.csv")
    original_dataset["promptCosDis"] = final["promptCosDis"]
    original_dataset["peerCosDis"] = final["peerCosDis"]
    
    # Read the previous results of LLM, and save them in the same table.
    llm_result_df = pd.read_csv("Data/llm_result_zeroshot.csv")
    original_dataset[f"{model.split('/')[-1]}-0s"] = llm_result_df[f"{model.split('/')[-1]}-0s"]
    
    # Apply rephrasing attacks and compute DSI of the attacked solutions.
    rephrase_model_list = ["openai/gpt-4o-2024-11-20",]
    with open("prompts/rephrase.txt", 'r') as file:
        rephrase_prompt_template = Template(file.read())
    question_list = [rephrase_prompt_template.substitute(text=solution) for solution in original_dataset["Solutions"]]
    # timestamp = test(question_list, temperature_list=[0.,], model_list=rephrase_model_list)
    timestamp = "20250915_202814"
    
    attacked_solution_list = []
    with open(f"tmp/test/{timestamp}/output.jsonl", 'r') as file:
        for line in file.readlines():
            attacked_solution_list.append(json.loads(line)["choices"][0]["message"]["content"])
    original_dataset["AttackedSolutions"] = attacked_solution_list

    model_name_list = ["bert", "bge", "Qwen"]
    dsi_name_list = ["AttackedDSI", "AttackedBgeDSI", "AttackedQwenDSI"]
    for model_name, dsi_name in zip(model_name_list, dsi_name_list):
        attacked_dsi_list = get_dsi(model_name, original_dataset["AttackedSolutions"])
        original_dataset[dsi_name] = attacked_dsi_list
    
    # Save the intermediate results.
    original_dataset.to_csv("Data/CPSTfulldataset2_rephrase_attacked.csv", index=False)

# Step 2: Embed the attacked solutions.
embedding_model_label_list = ["bert", "bge", "Qwen"]
embedding_model_list = ["bert-large-uncased", "BAAI/bge-large-en-v1.5", "Qwen/Qwen3-Embedding-0.6B"]
all_problem_IDs = sorted(original_dataset['ProblemID'].unique().tolist())

for embedding_model, embedding_model_label in zip(embedding_model_list, embedding_model_label_list):
    for problem_ID in all_problem_IDs:
        embedding_path = f"Data/embeddings/{embedding_model_label}/{problem_ID}/normalized_rephrase_attacked_text_embeddings.npy"
        if not os.path.exists(embedding_path):
            sub_dataset = original_dataset[original_dataset['ProblemID'] == problem_ID]
            solutions_list = sub_dataset['AttackedSolutions'].tolist()
            embed_solutions(
                problem_ID, 
                solutions_list, 
                embedding_model_label=embedding_model_label, 
                embedding_model=embedding_model, 
                save_path=f"Data/embeddings/{embedding_model_label}/{problem_ID}/normalized_rephrase_attacked_text_embeddings.npy"
            )

# Step 3: Calculate prompt & peer cosine distances.
embedding_model_label_list = ["bert", "bge", "Qwen"]

for embedding_model_label in embedding_model_label_list:
    original_dataset[f'AttackedPromptCosDis-{embedding_model_label}'] = float('nan')
    original_dataset[f'AttackedPeerCosDis-{embedding_model_label}'] = float('nan')

for embedding_model_label in embedding_model_label_list:
    for problem_ID in all_problem_IDs:
        embedding_directory = f"Data/embeddings/{embedding_model_label}/{problem_ID}/"
        
        # Load the original embeddings, and extract the query embedding (the first one).
        original_embeddings = np.load(os.path.join(embedding_directory, "normalized_text_embeddings.npy"))
        query_embedding = original_embeddings[0]
        
        # Load the attacked embeddings.
        attacked_embeddings = np.load(os.path.join(embedding_directory, "normalized_rephrase_attacked_text_embeddings.npy"))
        
        # Calculate prompt/peer cosine distances.
        prompt_cos_distances = get_prompt_distance_from_embeddings(query_embedding, attacked_embeddings)
        peer_cos_distances = get_peer_distance_from_embeddings(attacked_embeddings)
        
        # Store the results back to the original dataset.
        original_dataset.loc[original_dataset['ProblemID'] == problem_ID, f'AttackedPromptCosDis-{embedding_model_label}'] = prompt_cos_distances
        original_dataset.loc[original_dataset['ProblemID'] == problem_ID, f'AttackedPeerCosDis-{embedding_model_label}'] = peer_cos_distances
    
    # Organize a specific version for later performance analysis.
    final_df = pd.DataFrame(columns=["...1", "Solutions", "FacScoresQ", "FacScoresO", "Dataset", "ProblemID", "set", "ID", "wordcount", "DSI", "promptCosDis", "peerCosDis"])
    for column in final_df.columns:
        if column == "Solutions":
            final_df[column] = original_dataset["AttackedSolutions"]
        elif column == "DSI":
            final_df[column] = original_dataset["AttackedDSI"] if embedding_model_label == "bert" else original_dataset[f"Attacked{embedding_model_label.capitalize()}DSI"]
        elif "CosDis" in column:
            final_df[column] = original_dataset[f"Attacked{column[0].capitalize()+column[1:]}-{embedding_model_label}"]
        else:
            final_df[column] = original_dataset[column]
    final_df.to_csv(f"Results/{embedding_model_label}/final_rephrase_attacked.csv")
    
# Step 4: Test if the performance of LLM decreases on the attacked solutions.
with open(os.path.join("prompts/zero-shot.txt"), 'r') as file:
    prompt_template = Template(file.read())
    
problem_dict = {}
for problem_ID in all_problem_IDs:
    with open(f"Data/problem/{problem_ID}.txt", "r") as file:
        problem = file.read().strip()
    problem_dict[problem_ID] = problem

question_list = [
    prompt_template.substitute(problem=problem_dict[data.ProblemID], response=data.AttackedSolutions) 
for data in original_dataset.itertuples()]

# timestamp = test(question_list, temperature_list=[0,], model_list=[model,])
timestamp = "20250915_225518"
originality_scores = getOutputs("test", timestamp)
original_dataset[f"{model.split('/')[-1]}-0s-RephraseAttacked"] = originality_scores
    
# Step 5: Save the updated dataset with new metrics.
original_dataset.to_csv("Data/final_rephrase_attacked.csv", index=False, encoding="utf-8-sig")

# Step 6: Compare the performance of different embedding models on various sub-datasets 
# separated by problem ID.
sample_num = 20
np.random.seed(42)
seeds = np.random.randint(0, 10000, sample_num).tolist()
# seeds = [860]

param_comb = [["self"], ["self", "prompt"], ["self", "prompt", "peer"]]#[["self"], ["prompt"], ["peer"], ["self", "peer"], ["self", "prompt"], ["self", "peer", "prompt"]]

performance_metric = "CORR" # "CORR" or "MSE"
optimization_goal = "CORR" # "CORR" or "MSE"
example_num = 150
performanceDict = {}

for model_name in embedding_model_label_list:
    print("Current Model = {}".format(model_name))
    performanceDict[model_name] = {}
    full_dataset_path = f'Results/{model_name}/final_rephrase_attacked.csv'
    full_dataset = pd.read_csv(full_dataset_path, encoding="utf-8-sig")

    performance_values = []
    performance_std = []
    for problem in all_problem_IDs:
        print("Processing {}".format(problem))
        show_statistic(full_dataset[full_dataset["ProblemID"] == problem])
        performanceDict[model_name][problem] = {}
        
        all_values = [[] for _ in range(len(param_comb))]
        for seed in seeds:
            training_data, testing_data = get_data(
                full_dataset, problem_id=problem, example_num=example_num,
                dsi_type="DSI", prompt_type="CosDis", peer_type="CosDis",
                seed=seed, reallocate_testing_set=True
            )
            for idx, params in enumerate(param_comb):
                corr, mse = calculate_performance(training_data, testing_data, parameters=params, optimization_goal=optimization_goal)
                value = mse if performance_metric == "MSE" else corr
                all_values[idx].append(value)
                print(f"Seed {seed}, Params {params}: {performance_metric} = {value:.5f}")
                print("\n")
                
        # Save mean and std for each param_comb
        means = [np.mean(vals) for vals in all_values]
        stds = [np.std(vals) for vals in all_values]
        performance_values.append(means)
        performance_std.append(stds)
        
        for idx, params in enumerate(param_comb):
            performanceDict[model_name][problem][', '.join(params)] = {
                'mean': means[idx],
                'std': stds[idx]
            }
            
    performance_values = np.array(performance_values)
    performance_std = np.array(performance_std)
    
    draw(all_problem_IDs, performance_values, performance_std, model_name, example_num, sample_num, performance_metric, optimization_goal, Path(f"Results/{model_name}/rephrase_attacked"), param_comb, save_pic=True)

# Step 7: record the zero-shot performance of LLM on different sub-datasets and full dataset.
LLM_result_directory = "Results/LLM-zeroshot/rephrase_attacked/"
if not os.path.exists(LLM_result_directory):
    os.makedirs(LLM_result_directory)

with open(f"Results/LLM-zeroshot/rephrase_attacked/corr.txt", "w+") as file:
    for problem in all_problem_IDs:
        sub_dataset = original_dataset[original_dataset["ProblemID"] == problem]
        sub_human_scores = sub_dataset["FacScoresO"]
        sub_LLM_scores = sub_dataset[f"{model.split('/')[-1]}-0s-RephraseAttacked"]
        file.write(f"Corr(Human, LLM) on Problem ID {problem}: {np.corrcoef(sub_human_scores, sub_LLM_scores)[0,1]}\n")
    all_human_scores = original_dataset["FacScoresO"]
    all_LLM_scores = original_dataset[f"{model.split('/')[-1]}-0s-RephraseAttacked"]
    file.write(f"Corr(Human, LLM) on full dataset: {np.corrcoef(all_human_scores, all_LLM_scores)[0,1]}\n")
