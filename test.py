# 由于观测到预测结果的非单调性（例如Qwen模型在Joan数据集上的表现），我们希望做一个“过拟合”测试，检查线性模型是否是一个合理的模型


from turtle import clear
from utils.cosine_distance import get_prompt_distance, get_peer_distance
from utils.dsi import get_dsi
from utils.optimization import get_tuple, calculate_performance
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

from scipy.stats import pearsonr

def output_data(data, parameters):
    X, y = data[0], data[1]
    for i, item in enumerate(X):
        features = []
        if "self" in parameters:
            features.append(f"{float(item[0]):.3f}")
        if "prompt" in parameters:
            features.append(f"{float(item[1]):.3f}")
        if "peer" in parameters:
            features.append(f"{float(item[2]):.3f}")
        itemX = ",".join(features)
        print(f"({itemX}) -> {y[i]}")

    DSI = [item[0] for item in X]
    print("CORR = {}".format(pearsonr(y, DSI)))

def get_fulldata(full_dataset, problem_id, dsi_type = "DSI", prompt_type = "CosDis", peer_type = "CosDis"):
    items = full_dataset[full_dataset['ProblemID'] == problem_id]
    #items = full_dataset[(full_dataset['ProblemID'] == problem_id) & (full_dataset['set'] != 'training')]
    print(len(items))
    data = [get_tuple(row, dsi_type, prompt_type, peer_type) for row in items.itertuples()]
    X = np.array([item[:-1] for item in data])
    y = np.array([item[-1] for item in data])
    return (X, y)

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    label = None 
    
    models = ["Qwen"]
    sample_num = 5
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, sample_num).tolist()
    performance_metric = "CORR" # "CORR" or "MSE"

    example_num = 150
    for model_name in models:
        if label is None:
            file_path = script_dir / f'Results/{model_name}/'
        else:
            file_path = script_dir / f'Results/{label}/{model_name}/'
        full_dataset_path = file_path / 'final.csv'
        full_dataset = pd.read_csv(full_dataset_path, encoding="utf-8-sig")

        problem_ids = full_dataset["ProblemID"].unique()
        #problem_ids = ["Joan"]
        performance_values = []
        performance_std = []
        for problem in problem_ids:
            print("Processing {}".format(problem))
            values1, values2, values3 = [], [], []
            full_data = get_fulldata(full_dataset, problem_id=problem, dsi_type="DSI", prompt_type="CosDis", peer_type="CosDis")

            corr1, mse1 = calculate_performance(full_data, full_data, parameters=["self"], optimization_goal='MSE')
            values1.append(mse1 if performance_metric == "MSE" else corr1)
            # print(corr1, mse1)

            corr2, mse2 = calculate_performance(full_data, full_data, parameters=["self", "prompt"], optimization_goal='MSE')
            values2.append(mse2 if performance_metric == "MSE" else corr2)

            corr3, mse3 = calculate_performance(full_data, full_data, parameters=["self", "prompt", "peer"], optimization_goal='MSE')
            values3.append(mse3 if performance_metric == "MSE" else corr3)

            print("CORR in Full Data: self = {:.5f}, self+prompt = {:.5f}, self+prompt+peer = {:.5f}".format(values1[-1], values2[-1], values3[-1]))
            print("================================================")

            performance_values.append([np.mean(values1), np.mean(values2), np.mean(values3)])
            performance_std.append([np.std(values1), np.std(values2), np.std(values3)])
            print("\n")


        performance_values = np.array(performance_values)
        performance_std = np.array(performance_std)
        
        x = np.arange(len(problem_ids))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, performance_values[:, 0], width, label='self')
        ax.bar(x, performance_values[:, 1], width, label='self + prompt')
        ax.bar(x + width, performance_values[:, 2], width, label='self + prompt + peer')
        ax.errorbar(x - width, performance_values[:, 0], yerr=performance_std[:, 0], fmt='none', ecolor='black', capsize=5)
        ax.errorbar(x, performance_values[:, 1], yerr=performance_std[:, 1], fmt='none', ecolor='black', capsize=5)
        ax.errorbar(x + width, performance_values[:, 2], yerr=performance_std[:, 2], fmt='none', ecolor='black', capsize=5)
        # 在每个柱形上显示performance的值，保留3位小数
        for i in range(len(problem_ids)):
            ax.text(x[i] - width, performance_values[i, 0], f"{performance_values[i, 0]:.3f}", ha='center', va='bottom', fontsize=8)
            ax.text(x[i], performance_values[i, 1], f"{performance_values[i, 1]:.3f}", ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + width, performance_values[i, 2], f"{performance_values[i, 2]:.3f}", ha='center', va='bottom', fontsize=8)
        ax.legend() # 显示图例
        ax.set_ylabel('Performance')
        ax.set_title(f'{performance_metric} Comparison for Model: {model_name}(#Example={example_num})')
        ax.set_xticks(x)
        ax.set_xticklabels(problem_ids, ha='center')
        plt.tight_layout()
        plt.show()
        #save_path = file_path / "CORR-Comparison-{}-shot-1sample.jpg".format(example_num)
        #print(save_path)
        #plt.savefig(save_path) # 保存图片到指定路径
