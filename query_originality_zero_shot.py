import os
import json
import numpy as np
import pandas as pd

from string import Template

from utils.query import test


def QuestionGenerator(prompt_file):
    full_dataset = pd.read_csv('Data/CPSTfulldataset2.csv')
    
    for problem_ID in full_dataset['ProblemID'].unique():
        with open(f"Data/problem/{problem_ID}.txt", "r") as file:
            problem = file.read().strip()
        full_dataset.loc[full_dataset['ProblemID'] == problem_ID, 'Problem'] = problem

    with open(os.path.join(f"prompts/{prompt_file}.txt"), 'r') as file:
        prompt_template = Template(file.read())
    
    question_list = [
        prompt_template.substitute(problem=data.Problem, response=data.Solutions) 
    for data in full_dataset.itertuples()]
    
    return question_list, full_dataset


def getOutputs(path, timestamp, seed=0):
    folder_path = f"tmp/{path}/{timestamp}"
    output_file = os.path.join(folder_path, "output.jsonl")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        
    np.random.seed(seed)
    originality_scores = [
        int(json.loads(line)['choices'][0]['message']['content'].strip()[-1])
        if json.loads(line)['choices'][0]['message']['content'].strip()[-1].isdigit()
        else np.random.randint(5)
    for line in lines]
    
    return originality_scores


if __name__ == "__main__":
    model = "google/gemini-2.5-flash"
    questions, full_dataset = QuestionGenerator("zero-shot")
    timestamp = test(questions, temperature_list=[0,], model_list=[model,])
    # timestamp = "20250728_220035"
    originality_scores = getOutputs("test", timestamp)
    
    result_path = "Data/llm_result_zeroshot.csv"
    if os.path.exists(result_path):
        result_df = pd.read_csv(result_path)
        result_df.loc[:, f"{model.split('/')[-1]}-0s"] = originality_scores
        result_df.to_csv(result_path, index=False)
    else:
        full_dataset.loc[:, f"{model.split('/')[-1]}-0s"] = originality_scores
        sub_dataset = full_dataset[['ProblemID', 'Solutions', 'FacScoresO', 'set', f"{model.split('/')[-1]}-0s"]]
        sub_dataset.to_csv(result_path, index=False)
