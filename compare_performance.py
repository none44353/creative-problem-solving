
from turtle import clear
from utils.cosine_distance import get_prompt_distance, get_peer_distance
from utils.dsi import get_dsi
from utils.optimization import get_data, calculate_performance
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    label = None 
    
    #models = ["Qwen", "bert", "bge"]
    models = ["Qwen"]
    # for model_name in models:
    #     print(f"Processing model: {model_name}")
        
    #     df = pd.read_csv('Data/CPSTfulldataset2.csv', encoding="utf-8-sig") 
    #     documents = df['Solutions'].tolist() #如果是修改后的文本，需要从其他地方读如documents
    #     # 【重要】：假设embedding得结果已经事先存到f"Data/embeddings/{label}/{model_name}/{problem_id}/normalized_text_embeddings.npy"中

    #     # 代码目标：计算dsi 
    #     df['DSI'] = float('nan')
    #     dsi = get_dsi(model_name, documents)
    #     df['DSI'] = dsi
        
    #     # 代码目标：计算promptCosDis和peerCosDis，并将结果保存到final.csv中   
    #     df['promptCosDis'] = float('nan')
    #     df['peerCosDis'] = float('nan')
    #     problem_ids = df["ProblemID"].unique()
    #     for id in problem_ids:
    #         solutions_mask = (df['ProblemID'] == id)
    #         promptCosDis = get_prompt_distance(model_name, id, label)
    #         peerCosDis = get_peer_distance(model_name, id, label)
    #         # 使用布尔掩码进行批量赋值，而不是在循环中逐行赋值
    #         df.loc[solutions_mask, 'promptCosDis'] = promptCosDis
    #         df.loc[solutions_mask, 'peerCosDis'] = peerCosDis

    #     if label is None:
    #         save_directory = script_dir / f'Results/{model_name}/'
    #     else:
    #         save_directory = script_dir / f'Results/{label}/{model_name}/'
    #     file_name = 'final.csv'
    #     full_save_path = os.path.join(save_directory, file_name)
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #     df.to_csv(full_save_path, index=False, encoding="utf-8-sig")
    
    sample_num = 20
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, sample_num).tolist()
    performance_metric = "CORR" # "CORR" or "MSE"

    example_num = 50
    for model_name in models:
        if label is None:
            file_path = script_dir / f'Results/{model_name}/'
        else:
            file_path = script_dir / f'Results/{label}/{model_name}/'
        full_dataset_path = file_path / 'final.csv'
        full_dataset = pd.read_csv(full_dataset_path, encoding="utf-8-sig")

        problem_ids = full_dataset["ProblemID"].unique()
        #problem_ids = ["Clara"]
        performance_values = []
        performance_std = []
        for problem in problem_ids:
            print("Processing {}".format(problem))
            values1, values2, values3 = [], [], []
            for seed in seeds:
                training_data, testing_data = get_data(full_dataset, problem_id=problem, example_num=example_num, dsi_type="DSI", prompt_type="CosDis", peer_type="CosDis", seed=seed)
                print(len(testing_data[0]))
                corr1, mse1 = calculate_performance(training_data, testing_data, parameters=["self"], optimization_goal='MSE')
                print("Self Performance: MSE={}, CORR={}".format(mse1, corr1))
                values1.append(mse1 if performance_metric == "MSE" else corr1)

                corr2, mse2 = calculate_performance(training_data, testing_data, parameters=["self", "prompt"], optimization_goal='MSE')
                print("Self + Prompt Performance: MSE={}, CORR={}".format(mse2, corr2))
                values2.append(mse2 if performance_metric == "MSE" else corr2)

                corr3, mse3 = calculate_performance(training_data, testing_data, parameters=["self", "prompt", "peer"], optimization_goal='MSE')
                print("Self + Prompt + Peer Performance: MSE={}, CORR={}".format(mse3, corr3))
                values3.append(mse3 if performance_metric == "MSE" else corr3)
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
