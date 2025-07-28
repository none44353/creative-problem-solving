import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from utils.calculate_dist import get_semantic_similarity, get_peer_distance


def calculate_correlation_after_outlier_removal(y_var, x_var, cutoff_val):
    y_var = np.asarray(y_var)
    x_var = np.asarray(x_var)

    X = sm.add_constant(x_var)
    # print(X[:3], y_var[:3])
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


def correlation_with_k_peers(k_values, solutions_list, ID, fac_scores, cutoff):
    distance1 = get_semantic_similarity(ID, mode="cosine_distance")
    correlations = []
    
    for k in k_values:
        distance2 = get_peer_distance(solutions_list=solutions_list, mode="cosine_distance", number_of_peers=k)
        
        lam = 0.6
        preds = [(1 - lam) * d1 + lam * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is cosine_distance
    
        preds = np.array(preds)
        preds = 4 * (preds - preds.min()) / (preds.max() - preds.min()) if preds.max() > preds.min() else np.zeros_like(preds)
        
        pearson_corr = calculate_correlation_after_outlier_removal(y_var=fac_scores, x_var=preds, cutoff_val=cutoff)
        correlations.append(pearson_corr)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, correlations, marker='o', linestyle='-')
    plt.xlabel('k_values')
    plt.ylabel('Pearson Correlation')
    plt.title('Correlation vs. Number of Peers (k)')
    plt.grid(True)
    for i, (k, corr) in enumerate(zip(k_values, correlations)):
        plt.annotate(f'k={k}\n{corr:.2f}', (k, correlations[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.show()
    
    return correlations


def compute_correlation_with_filter(df, filter_condition, prediction_column, metric_column='FacScoresO'):
    # 示例用法
    # corr = compute_correlation_with_filter(df, "ProblemID == 'Becky' and FacScoresO > 2", metric_column='FacScoresO')
    # print(f"相关系数: {corr}")
    filtered_df = df.query(filter_condition)
    predictions = filtered_df[prediction_column]
    metric = filtered_df[metric_column]
    
    print(filter_condition, len(filtered_df), len(predictions))
    
    cutoff = 4 / (len(filtered_df) - 2) if len(filtered_df) > 2 else 0.1
    
    pearson_corr = calculate_correlation_after_outlier_removal(y_var=metric, x_var=predictions, cutoff_val=cutoff)
    
    return pearson_corr
