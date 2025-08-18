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
    
def compute_correlation(dataset, x_name, y_name = 'FacScoresO'):
    # 示例用法
    # corr = compute_correlation_with_filter(df, "ProblemID == 'Becky' and FacScoresO > 2", metric_column='FacScoresO')
    # print(f"相关系数: {corr}")
    x = dataset[x_name]
    y = dataset[y_name]
    cutoff = 4 / (len(y) - 2) if len(y) > 2 else 0.1
    correlation = calculate_correlation_after_outlier_removal(y_var=y, x_var=x, cutoff_val=cutoff)
    print(f"Correlation between {x_name} and {y_name} (after outlier removal): {correlation}")
    

if __name__ == "__main__":
    full_dataset = pd.read_csv('Data/final.csv', encoding= "utf-8-sig")

    compute_correlation(full_dataset, x_name='DSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='QwenDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='bgeDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='wordQwenDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='wordbgeDSI', y_name='FacScoresO')