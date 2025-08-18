import os
import json
import pygad
import numpy as np
import pandas as pd
from string import Template
from utils.query import test
from statistics import correlation
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import matplotlib.pyplot as plt


def get_tuple(data, dsi_type, prompt_type, peer_type):
    if dsi_type == "DSI":
        dsi_value = data.DSI
    elif dsi_type == "QwenDSI":
        dsi_value = data.QwenDSI
    elif dsi_type == "bgeDSI":
        dsi_value = data.bgeDSI
    elif dsi_type == "wordQwenDSI":
        dsi_value = data.wordQwenDSI
    elif dsi_type == "wordbgeDSI":
        dsi_value = data.wordbgeDSI

    if prompt_type == "DSI":
        prompt_value = data.promptDSI
    elif prompt_type == "cosDis":
        prompt_value = data.promptCosDis

    if peer_type == "DSI":
        peer_value = data.peerDSI
    elif peer_type == "cosDis":
        peer_value = data.peerCosDis

    return (dsi_value, prompt_value, peer_value, data.FacScoresO)
    

def ExampleGenerator(full_dataset, problemID, example_num, dsi_type, prompt_type, peer_type, seed=0):
    assert isinstance(example_num, int) and example_num > 0, "example_num must be a positive integer"
    train_dataset = full_dataset[full_dataset['set'] == 'training']
    
    if problemID != "Mike":
        example_items = train_dataset[train_dataset['ProblemID'] == problemID].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]
    elif problemID == "Mike":
        selected_problem_id = train_dataset['ProblemID'].sample(n=1, random_state=seed).item()
        example_items = train_dataset[train_dataset['ProblemID'] == selected_problem_id].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]

    return examples


def calculate_correlation_after_outlier_removal(y_var, x_var, cutoff_val):
    y_var = np.asarray(y_var)
    x_var = np.asarray(x_var)

    X = sm.add_constant(x_var)
    model = sm.OLS(y_var, X)
    fit_model = model.fit()
    cooks_dist = fit_model.get_influence().cooks_distance[0]

    # 找出 Cook's Distance 超过截断值的索引
    above_cutoff_indices = np.where(cooks_dist > cutoff_val)[0]

    # 移除异常值
    if len(above_cutoff_indices) > 0:
        x_cleaned = np.delete(x_var, above_cutoff_indices)
        y_cleaned = np.delete(y_var, above_cutoff_indices)
    else:
        x_cleaned = x_var
        y_cleaned = y_var     
    #print(f"Number of outliers removed: {len(above_cutoff_indices)}")

    # 计算移除异常值后的 Pearson 相关系数
    if len(x_cleaned) > 1:
        correlation_after_removal, _ = pearsonr(y_cleaned, x_cleaned)
    else:
        correlation_after_removal = np.nan # 数据点不足，无法计算相关性

    # 计算预测误差和相关统计量
    df_cleaned = pd.DataFrame({'label': y_cleaned, 'prediction': x_cleaned})

    mean_value = np.mean(df_cleaned['label'])
    sd_value = np.std(df_cleaned['label'])
    lower_threshold = mean_value - 2 * sd_value
    upper_threshold = mean_value + 2 * sd_value

    selected_values_low = df_cleaned['prediction'][df_cleaned['label'] < lower_threshold]
    selected_values_up = df_cleaned['prediction'][df_cleaned['label'] > upper_threshold]

    mean_prediction_extreme_labels = np.mean(np.abs(np.concatenate((selected_values_up, selected_values_low)))) if len(selected_values_up) + len(selected_values_low) > 0 else np.nan

    mid_labels_mask = (df_cleaned['label'] >= lower_threshold) & (df_cleaned['label'] <= upper_threshold)
    mean_prediction_mid_labels = np.mean(np.abs(df_cleaned['prediction'][mid_labels_mask])) if np.any(mid_labels_mask) else np.nan

    return correlation_after_removal

# 用进化算法直接优化预测模型在training data上的correlation
def fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes):
    if num_genes == 3:
        # 只使用 lambda1, lambda2, lambda3
        y_pred = np.dot(X_train, solution)
    else:
        y_pred = np.dot(X_train, solution[:-1]) + solution[-1]

    cutoff = 4 / (len(y_train) - 2) if len(y_train) > 2 else 0.1
    correlation_value = calculate_correlation_after_outlier_removal(y_train, y_pred, cutoff)

    return correlation_value

# 假设这是用于运行进化算法的函数
def genetic_algorithm(X_train, y_train, num_genes):

    fitness_function = lambda ga_instance, solution, solution_idx: fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes)

    ga_instance = pygad.GA(
        num_generations=200,
        num_parents_mating=20,
        fitness_func=fitness_function,
        sol_per_pop=40,
        num_genes=num_genes,  # 对应 lambda1, lambda2, lambda3, lambda4
        parent_selection_type="rws", #rws: 适应度越高的个体被选中的概率越大，sss: 选择适应度最高的个体
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes="default"
    )

    ga_instance.run()
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
    return best_solution


def get_predictions(full_dataset, dsi_type, prompt_type, peer_type, num, problem_list, optimization_goal,  with_lambda4 = True):
    data_with_prediction = full_dataset.copy()
    data_with_prediction[f'prediction'] = np.nan

    for problem in problem_list:
        examples = ExampleGenerator(full_dataset, problem, dsi_type=dsi_type, prompt_type=prompt_type, peer_type=peer_type, example_num=num)
        X_train = np.array([item[:-1] for item in examples])
        y_train = np.array([item[-1] for item in examples])

        test_items = data_with_prediction[(data_with_prediction['ProblemID'] == problem) & (data_with_prediction['set'] != 'training')]
        X_test = np.array([get_tuple(row, dsi_type, prompt_type, peer_type)[:-1] for row in test_items.itertuples()])

        if optimization_goal == 'MSE':

            model = LinearRegression(fit_intercept=with_lambda4)
            model.fit(X_train, y_train)

            print(f"Problem: {problem}, Type: {dsi_type}+{prompt_type}+{peer_type}, Example Num: {num}")
            print("Model coefficients:", model.coef_)
            print("Model intercept:", model.intercept_)

            prediction = model.predict(X_test)
            
        elif optimization_goal == 'correlation':
            solution = genetic_algorithm(X_train, y_train, num_genes=4 if with_lambda4 else 3)
            
            print(f"Problem: {problem}, Type: {dsi_type}+{prompt_type}+{peer_type}, Example Num: {num}")
            print("Model coefficients:", solution)
            if with_lambda4:
                prediction = np.dot(X_test, solution[:-1]) + solution[-1]
            else:
                prediction = np.dot(X_test, solution)

        prediction_scaled = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)

        # 预测取整
        # prediction_scaled = np.round(prediction_scaled).astype(int)

        print("Predictions (first 5):", prediction_scaled[:5], "\n")

        # 使用 test_items 的索引进行赋值
        data_with_prediction.loc[test_items.index, f'prediction'] = prediction_scaled

    return data_with_prediction

def compute_correlation_with_filter(df, filter_condition, prediction_column, metric_column='FacScoresO'):
    # 示例用法
    # corr = compute_correlation_with_filter(df, "ProblemID == 'Becky' and FacScoresO > 2", metric_column='FacScoresO')
    # print(f"相关系数: {corr}")
    filtered_df = df.query(filter_condition)
    predictions = filtered_df[prediction_column]
    metric = filtered_df[metric_column]
    
    print(f"filter_condition: {filter_condition} + Number of predictions: {len(predictions)}")
    
    cutoff = 4 / (len(filtered_df) - 2) if len(filtered_df) > 2 else 0.1
    
    pearson_corr = calculate_correlation_after_outlier_removal(y_var=metric, x_var=predictions, cutoff_val=cutoff)
    
    return pearson_corr

if __name__ == "__main__":
    problem_list= ["Acme", "Becky", "Clara", "Joan", "Mike", "Ralph"]
    example_num_list = [5, 10, 20]
    
    # Load the full dataset and remap the originality score to a 4-point scale 
    # according to the splitted quartiles.
    full_dataset = pd.read_csv('Data/final.csv', encoding= "utf-8-sig")
    full_dataset['QuartilesO'] = pd.qcut(full_dataset['FacScoresO'], 4, labels=False)    
    
    set_names = problem_list + ['test', 'heldout', 'all']
    
    # 保存每个 type 的 correlation 结果到不同的 sheet
    with pd.ExcelWriter('Data/Ours/CORR-result.xlsx', engine='openpyxl') as writer:
        for dsi_type in ["DSI", "bgeDSI", "QwenDSI", "wordbgeDSI", "wordQwenDSI"]:
            prompt_type = "cosDis"
            peer_type = "cosDis"
            
            correlation_table = []
            for num in example_num_list:
                data_with_prediction = get_predictions(full_dataset, dsi_type=dsi_type, prompt_type=prompt_type, peer_type=peer_type, num=num, problem_list=problem_list, optimization_goal="correlation")

                row = []
                for set_name in set_names:
                    if set_name == 'all':
                        filter_cond = "set != 'training'"
                    elif set_name in problem_list:
                        filter_cond = f"ProblemID == '{set_name}' and set != 'training'"
                    else:
                        filter_cond = f"set == '{set_name}'"
                    corr = compute_correlation_with_filter(data_with_prediction, filter_cond, prediction_column='prediction', metric_column='FacScoresO')
                    row.append(corr)
                correlation_table.append(row)    
                    
            lambda_correlation_df = pd.DataFrame(correlation_table, columns=set_names)
            lambda_correlation_df.insert(0, 'example_num', example_num_list)
            
            def highlight_max(s):
                is_max = s == s.max()
                return ['color: red' if v else '' for v in is_max]

            styler = lambda_correlation_df.style.apply(highlight_max, axis=0, subset=lambda_correlation_df.columns[1:])
            styler.to_excel(writer, sheet_name=f"{dsi_type}+{prompt_type}+{peer_type}", index=False)
    print("Excel with multiple sheets saved.")
    