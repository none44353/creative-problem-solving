import pandas as pd

from utils.calculate_corr import compute_correlation_with_filter


llm_result_df = pd.read_csv("Data/llm_result.csv")
prediction_cols = llm_result_df.columns[4:].tolist()
problem_list = sorted(llm_result_df['ProblemID'].unique().tolist())
set_names = problem_list + ['test', 'heldout', 'all']

for prediction_col in prediction_cols:
    print(f"Current processing: {prediction_col}")
    
    corr_list = []
    for set_name in set_names:
        if set_name == 'all':
            filter_cond = "index >= 0"
        elif set_name in problem_list:
            filter_cond = f"ProblemID == '{set_name}' & set == 'test'" if set_name != 'Mike' else f"ProblemID == '{set_name}'"
        else:
            filter_cond = f"set == '{set_name}'"
        corr = compute_correlation_with_filter(
            llm_result_df, filter_cond, prediction_col, metric_column='FacScoresO'
        ).item()
        corr_list.append(corr)
        # print(f"Correlation of {set_name}: {corr}")
    print(" ".join(list(map(str, corr_list))))
