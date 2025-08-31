import os
import json
import numpy as np
import pandas as pd
from utils.calculate_corr import calculate_correlation_after_outlier_removal
# from statistics import correlation
# from scipy.stats import pearsonr
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics.pairwise import cosine_similarity
# import statsmodels.api as sm
# import matplotlib.pyplot as plt


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
    full_dataset = pd.read_csv('Data/final-backup.csv', encoding= "utf-8-sig")

    compute_correlation(full_dataset, x_name='DSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='QwenDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='bgeDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='wordQwenDSI', y_name='FacScoresO')
    compute_correlation(full_dataset, x_name='wordbgeDSI', y_name='FacScoresO')