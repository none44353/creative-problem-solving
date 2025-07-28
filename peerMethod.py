import pandas as pd

from utils.calculate_corr import compute_correlation_with_filter
from utils.calculate_dist import get_semantic_similarity, get_peer_distance


def build_df(problem_list, lambdas):
    df = pd.read_csv('Data/CPSTfulldataset2.csv')

    for ID in problem_list:
        solutions = df[df['ProblemID'] == ID]['Solutions']
        solutions_list = solutions.tolist()
        print(f"Processing ProblemID: {ID} with {len(solutions_list)} solutions.")

        distance1 = get_semantic_similarity(ID, mode="DSI")
        distance2 = get_peer_distance(ID, solutions_list=solutions_list, mode="cosine_distance")
        for lamb in lambdas:
            #prediction = [(1 - lamb) * d1 - lamb * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is Jaccard
            prediction = [(1 - lamb) * d1 + lamb * d2 for d1, d2 in zip(distance1, distance2)] # when distance2 is cosine_distance
            
            # prediction = np.array(prediction)
            # prediction = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)

            df.loc[df['ProblemID'] == ID, f'prediction_{lamb}'] = prediction

    return df


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
                filter_cond = f"ProblemID == '{set_name}'"
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
    lambda_correlation_df.to_csv('Data/Ours/result-DSI.csv', index=False)
    print("Lambda correlation results saved.")


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