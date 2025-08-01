import os
import json
import numpy as np
import pandas as pd

from string import Template

from utils.query import test


def QuestionGenerator(prompt_file, example_num, seed=0):
    assert isinstance(example_num, int) and example_num > 0, "example_num must be a positive integer"
    
    # Load the full dataset and remap the originality score to a 4-point scale 
    # according to the splitted quartiles.
    full_dataset = pd.read_csv('Data/CPSTfulldataset2.csv')
    full_dataset['QuartilesO'] = pd.qcut(full_dataset['FacScoresO'], 4, labels=False)
    
    for problem_ID in full_dataset['ProblemID'].unique():
        with open(f"Data/problem/{problem_ID}.txt", "r") as file:
            problem = file.read().strip()
        full_dataset.loc[full_dataset['ProblemID'] == problem_ID, 'Problem'] = problem
        
    train_dataset = full_dataset[full_dataset['set'] == 'training']
    heldout_dataset = full_dataset[full_dataset['set'] != 'training']

    with open(os.path.join(f"prompts/{prompt_file}.txt"), 'r') as file:
        prompt_template = Template(file.read())
    
    question_list = []
    for data in heldout_dataset.itertuples():
        if data.set == "test":
            example_items = train_dataset[train_dataset['ProblemID'] == data.ProblemID].sample(n=example_num, random_state=seed)
            examples = "\n\n".join([f"{example.Solutions}\nRating: {example.QuartilesO}" for example in example_items.itertuples()])
        elif data.set == "heldout":
            selected_problem_id = train_dataset['ProblemID'].sample(n=1, random_state=seed).item()
            example_items = train_dataset[train_dataset['ProblemID'] == selected_problem_id].sample(n=example_num, random_state=seed)
            examples = f"Problem: {example_items.iloc[0]['Problem']}\n\n" + "\n\n".join([f"Solution: {example.Solutions}\nRating: {example.QuartilesO}" for example in example_items.itertuples()])
        question = prompt_template.substitute(problem=data.Problem, examples=examples, response=data.Solutions)
        question_list.append(question)
    
    return question_list, heldout_dataset


def getOutputs(path, timestamp, seed=0):
    folder_path = f"tmp/{path}/{timestamp}"
    output_file = os.path.join(folder_path, "output.jsonl")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        
    np.random.seed(seed)
    originality_scores = [
        int(json.loads(line)['choices'][0]['message']['content'].strip()[0])
        if json.loads(line)['choices'][0]['message']['content'].strip()[0].isdigit()
        else np.random.randint(4)
    for line in lines]
    
    return originality_scores


if __name__ == "__main__":
    for example_num in (10, 20):
        questions, heldout_dataset = QuestionGenerator("few-shot", example_num=example_num)
        print(questions[-1], end='\n\n')
        timestamp = test(questions, temperature_list=[0,])
        # timestamp = "20250731_100014"
        originality_scores = getOutputs("test", timestamp)
        
        result_path = "Data/llm_result_fewshot.csv"
        if os.path.exists(result_path):
            result_df = pd.read_csv(result_path)
            result_df.loc[:, f'gpt-4-turbo-{example_num}s'] = originality_scores
            result_df.to_csv(result_path, index=False)
        else:
            heldout_dataset.loc[:, f'gpt-4-turbo-{example_num}s'] = originality_scores
            sub_dataset = heldout_dataset[['ProblemID', 'Solutions', 'FacScoresO', 'set', f'gpt-4-turbo-{example_num}s']]
            sub_dataset.to_csv(result_path, index=False)
