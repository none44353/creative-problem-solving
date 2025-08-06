from statistics import correlation
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

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


def Jaccard(A, B):
    set_A = set(str(A).split())
    set_B = set(str(B).split())
    intersection = set_A & set_B
    union = set_A | set_B
    if not union:
        return 0.0
    return len(intersection) / len(union)

def get_embeddings(ID):
    #model: "Qwen3-Embedding-0.6B"
    full_save_path = f"Data/embeddings/{ID}/normalized_text_embeddings.npy"
    loaded_embeddings = np.load(full_save_path)
    
    query_embedding = loaded_embeddings[0]
    solutions_embeddings = loaded_embeddings[1:]
    return query_embedding, solutions_embeddings

def get_sementic_similarity(ID, mode = "DSI"):
    if mode == "cosine_distance":
        query_embedding, solutions_embeddings = get_embeddings(ID)
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            solutions_embeddings
        )[0]

        return similarities.tolist()
    
    elif mode == "DSI":
        df = pd.read_csv('Data/final.csv', encoding= "utf-8-sig")
        dsi_values = df[df['ProblemID'] == ID]['DSI'].tolist()
        return dsi_values
    
    elif mode == "prompt-DSI":
        df = pd.read_csv('Data/final.csv',  encoding= "utf-8-sig")
        prompt_dsi_values = df[df['ProblemID'] == ID]['prompt-DSI'].tolist()
        return prompt_dsi_values

def get_peer_distance(ID, solutions_list, mode, number_of_peers = 0):
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
    
    elif mode == "DSI":
        df = pd.read_csv('Data/final.csv',  encoding= "utf-8-sig")
        dsi_values = df[df['ProblemID'] == ID]['peer-DSI'].tolist()
        return dsi_values

def correlation_with_k_peers(k_values, solutions_list, ID, fac_scores, cutoff):
    distance1 = get_sementic_similarity(ID, mode="cosine_distance")
    correlations = []
    
    for k in k_values:
        distance2 = get_peer_distance(ID, solutions_list=solutions_list, mode="cosine_distance", number_of_peers=k)
        
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

def build_df(problem_list, lambdas):
    df = pd.read_csv('Data/CPSTfulldataset2.csv')

    for ID in problem_list:
        solutions = df[df['ProblemID'] == ID]['Solutions']
        solutions_list = solutions.tolist()
        print(f"Processing ProblemID: {ID} with {len(solutions_list)} solutions.")

        distance1 = get_sementic_similarity(ID, mode="cosine_distance")
        distance2 = get_peer_distance(ID, solutions_list=solutions_list, mode="DSI")
        for lamb in lambdas:
            #prediction = [(1 - lamb) * d1 - lamb * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is Jaccard
            prediction = [(1 - lamb) * d1 + lamb * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is cosine_distance
            
            # prediction = np.array(prediction)
            # prediction = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)

            df.loc[df['ProblemID'] == ID, f'prediction_{lamb}'] = prediction

    return df

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
    problem_list = ["Acme", "Becky", "Clara", "Joan", "Mike", "Ralph"]
    lambdas = [round(x * 0.1, 1) for x in range(11)]
    df = build_df(problem_list, lambdas)

    # 针对每个lambda值都计算相关系数，保存到表格
    set_names = problem_list + ['test', 'heldout', 'all']
    correlation_table = []

    for lamb in lambdas:
        row = []
        prediction_col = f'prediction_{lamb}'
        for set_name in set_names:
            if set_name == 'all':
                filter_cond = "index >= 0"
            elif set_name in problem_list:
                filter_cond = f"ProblemID == '{set_name}' and set != 'training'"
            else:
                filter_cond = f"set == '{set_name}'"
            corr = compute_correlation_with_filter(df, filter_cond, prediction_col, metric_column='FacScoresO')
            # 如果你想计算整个df的相关系数，可以将filter_cond设置为"index >= 0"
            # 例如：
            # filter_cond = "index >= 0"
            row.append(corr)
        correlation_table.append(row)

    lambda_correlation_df = pd.DataFrame(correlation_table, columns=set_names)
    lambda_correlation_df.insert(0, 'lambda', lambdas)
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['color: red' if v else '' for v in is_max]

    styled_df = lambda_correlation_df.style.apply(highlight_max, axis=0)
    styled_df.to_excel('Data/Ours/result-promptcosD+DSI.xlsx', index=False, engine='openpyxl')
    print("Colored Excel with max values saved.")


# fac_scores = df[df['ProblemID'] == ID]['FacScoresO'].tolist()
# lambdas = [round(x * 0.1, 1) for x in range(11)]
# predictions = []
# correlations = []
# predictions = {}
# for lam in lambdas:
#     #preds = [d1 - lam * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is Jaccard
#     preds = [(1 - lam) * d1 + lam * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is cosine_distance
    
#     preds = np.array(preds)
#     preds = 4 * (preds - preds.min()) / (preds.max() - preds.min()) if preds.max() > preds.min() else np.zeros_like(preds)
    
#     predictions[lam] = preds
    
#     pearson_corr = calculate_correlation_after_outlier_removal(y_var=fac_scores, x_var=preds, cutoff_val=cutoff)
#     correlations.append(pearson_corr)

# for idx, lam in enumerate(lambdas):
#     print(f"Lambda={lam}: Pearson correlation={correlations[idx]}")



# k_values = [5,10,20, 30, 50, 100, 200]
# correlations = correlation_with_k_peers(k_values=k_values, solutions_list=solutions_list, ID=ID, fac_scores=fac_scores, cutoff=cutoff)


# # save results to CSV
# output_rows = []
# for i, sol in enumerate(solutions_list):
#     row = [sol] + [predictions[lam][i] for lam in lambdas] + [fac_scores[i]]
#     output_rows.append(row)


# columns = ['solutions'] + [f'pred_{int(lam * 10)}' for lam in lambdas] + ['fac_scores']
# output_df = pd.DataFrame(output_rows, columns=columns)

# output_directory = f'Data/Ours/{ID}'
# output_file_path = f'{output_directory}/result.csv'

# # Check if the directory exists, and if not, create it
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
#     print(f"Created directory: {output_directory}")

# # Save the DataFrame to a CSV file
# output_df.to_csv(output_file_path, index=False)
# print(f"Data saved successfully to: {output_file_path}")