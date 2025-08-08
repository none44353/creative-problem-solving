import os
import json
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

def get_tuple(data, type):
    if type == "DSI-DSI":
        return (data.DSI, data.promptDSI, data.peerDSI, data.FacScoresO)
    elif type == "DSI-cosDis":
        return (data.DSI, data.promptDSI, data.peerCosDis, data.FacScoresO)
    elif type == "cosDis-DSI":
        return (data.DSI, data.promptCosDis, data.peerDSI, data.FacScoresO)
    elif type == "cosDis-cosDis":
        return (data.DSI, data.promptCosDis, data.peerCosDis, data.FacScoresO)
    

def ExampleGenerator(full_dataset, problemID, type, example_num, seed=0):
    assert isinstance(example_num, int) and example_num > 0, "example_num must be a positive integer"
    train_dataset = full_dataset[full_dataset['set'] == 'training']
    
    if problemID != "Mike":
        example_items = train_dataset[train_dataset['ProblemID'] == problemID].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, type) for example in example_items.itertuples()]
    elif problemID == "Mike":
        selected_problem_id = train_dataset['ProblemID'].sample(n=1, random_state=seed).item()
        example_items = train_dataset[train_dataset['ProblemID'] == selected_problem_id].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, type) for example in example_items.itertuples()]

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


if __name__ == "__main__":
    problem_list= ["Acme", "Becky", "Clara", "Joan", "Mike", "Ralph"]
    example_num_list = [5, 10, 20]
    type_list = ["DSI-DSI", "DSI-cosDis", "cosDis-DSI", "cosDis-cosDis"]    
    
    # Load the full dataset and remap the originality score to a 4-point scale 
    # according to the splitted quartiles.
    full_dataset = pd.read_csv('Data/final.csv', encoding= "utf-8-sig")
    full_dataset['QuartilesO'] = pd.qcut(full_dataset['FacScoresO'], 4, labels=False)    
    
    # 保存每个 type 的 correlation 结果到不同的 sheet
    with pd.ExcelWriter('Data/Ours/result-3parameters.xlsx', engine='openpyxl') as writer:
        for type_name in type_list:
            correlation_table = []
            for num in example_num_list:
                row = []
                for problem in problem_list:
                    examples = ExampleGenerator(full_dataset, problem, type=type_name, example_num=num)
                    X_train = np.array([item[:-1] for item in examples])
                    y_train = np.array([item[-1] for item in examples])

                    model = LinearRegression(fit_intercept=False)
                    model.fit(X_train, y_train)
                    
                    print(f"Problem: {problem}, Type: {type_name}, Example Num: {num}")
                    print("Model coefficients:", model.coef_)
                    print("Model intercept:", model.intercept_)

                    test_items = full_dataset[(full_dataset['set'] != 'training') & (full_dataset['ProblemID'] == problem)]
                    X_test = np.array([get_tuple(row, type_name)[:-1] for row in test_items.itertuples()])
                    y_test = np.array([row.FacScoresO for row in test_items.itertuples()])
                    prediction = model.predict(X_test)
                    
                    prediction = np.array(prediction)
                    prediction = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)

                    print("Predictions (first 5):", prediction[:5], "\n")
                    
                    cutoff = 4 / (len(y_test) - 2) if len(y_test) > 2 else 0.1
                    pearson_corr = calculate_correlation_after_outlier_removal(y_var=y_test, x_var=prediction, cutoff_val=cutoff)
                    row.append(pearson_corr)
                correlation_table.append(row)
            lambda_correlation_df = pd.DataFrame(correlation_table, columns=problem_list)
            lambda_correlation_df.insert(0, 'example_num', example_num_list)
            
            
                    
            def highlight_max(s):
                is_max = s == s.max()
                return ['color: red' if v else '' for v in is_max]

            styler = lambda_correlation_df.style.apply(highlight_max, axis=0, subset=lambda_correlation_df.columns[1:])
            styler.to_excel(writer, sheet_name=type_name, index=False)
    print("Excel with multiple sheets saved.")
    