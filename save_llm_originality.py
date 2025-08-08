import pandas as pd

from utils.calculate_corr import compute_correlation_with_filter

corr_result_df = pd.DataFrame()
csv_file_list = ["Data/llm_result_zeroshot.csv", "Data/llm_result_fewshot.csv"]

for csv_file in csv_file_list:
    llm_result_df = pd.read_csv(csv_file)
    prediction_cols = llm_result_df.columns[4:].tolist()
    problem_list = sorted(llm_result_df['ProblemID'].unique().tolist())
    set_names = problem_list + ['test', 'heldout', 'all']

    for prediction_col in prediction_cols:
        print(f"Current processing: {prediction_col}")

        corr_list = []
        for set_name in set_names:
            if set_name == 'all':
                filter_cond = "set != 'training'"
            elif set_name in problem_list:
                filter_cond = f"ProblemID == '{set_name}' & set != 'training'"
            else:
                filter_cond = f"set == '{set_name}'"
            corr = compute_correlation_with_filter(
                llm_result_df, filter_cond, prediction_col, metric_column='FacScoresO'
            ).item()
            corr_list.append(corr)
            corr_result_df.loc[prediction_col, set_name] = corr
            # print(f"Correlation of {set_name}: {corr}")
        print(" ".join(list(map(str, corr_list))))
        print()

corr_result_df.loc[sorted(corr_result_df.index.tolist(), key=lambda name: (name.split('-')[:-1], int(name.split('-')[-1][:-1])))].to_csv("Data/llm_result_corr.csv")