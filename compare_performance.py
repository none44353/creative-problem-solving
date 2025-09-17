
import colorsys
from turtle import clear, dot
from utils.cosine_distance import get_prompt_distance, get_peer_distance
from utils.dsi import get_dsi
from utils.calculate_corr import show_statistic
from utils.optimization import get_data, calculate_performance
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

from scipy.stats import pearsonr
from matplotlib.lines import Line2D

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

def str_trans(param):
    if param == "self":
        return "DSI"
    elif param == "prompt":
        return "PromptCosDis"
    elif param == "peer":
        return "PeerCosDis"
    else:
        raise ValueError("Unsupported parameter: {}".format(param))

def draw(problem_ids, performance_values, performance_std, model_name, example_num, sample_num, performance_metric, optimization_goal, file_path, param_comb, save_pic=True):
    x = np.arange(len(problem_ids))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Predictor Parameters: ' + ', '.join(str_trans(param) for param in params) for params in param_comb]
    colors = ['#c88d8d', '#8dc88d', '#8d8dc8']
    
    for idx, label in enumerate(labels):
        ax.bar(x + (idx - 1) * width, performance_values[:, idx], width, label=label, color=colors[idx])
        ax.errorbar(x + (idx - 1) * width, performance_values[:, idx], yerr=performance_std[:, idx], fmt='none', ecolor='black', capsize=5)
        for i in range(len(problem_ids)):
            ax.text(x[i] + (idx - 1) * width, performance_values[i, idx], f"{performance_values[i, idx]:.3f}", ha='center', va='bottom', fontsize=8)

    ax.legend() # 显示图例
    ax.set_ylabel('Performance')
    ax.set_title(f'{performance_metric} Comparison (Model: {model_name}, #Example: {example_num})')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_ids, ha='center')
    plt.tight_layout()
    #plt.show()
    
    if save_pic:
        if optimization_goal == 'CORR':
            file_path = file_path / 'CORR-Maximize/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        save_path = file_path / "Self-Prompt-Peer:CORR-Comparison-{}-shot-{}-sample.jpg".format(example_num, sample_num)
        print(save_path)
        plt.savefig(save_path) # 保存图片到指定路径

def drawLinePic(problem_ids, models, param_comb, performanceDict, save_pic=True):
    '''绘制一张折线图, 横坐标对应不同的param_comb, 纵坐标对应performanceDict中的‘mean’.
    每个problem_id对应一种颜色，每个model对应一种线型'''
    plt.figure(figsize=(10, 6))
    print("param_comb:", param_comb)
    x = [' + '.join(params) for params in param_comb]
    colors = plt.cm.get_cmap('tab10', len(problem_ids)).colors  # 获取足够的颜色
    linestyles = ['-', '--', '-.', ':']
    dotstyles = ['o', 's', '^', 'D', 'v', 'P', '*']  # 不同的点样式
   
    for i, model_name in enumerate(models):
        for j, problem_id in enumerate(problem_ids):
            means = []
            for params in param_comb:
                if problem_id in performanceDict[model_name]:
                    value = float(performanceDict[model_name][problem_id][', '.join(params)]['mean'])
                    means.append(value)
                    print(f"{model_name} - {problem_id} - {params}: {value}")
                else:
                    means.append(0)
            plt.plot(x, means, label=f"{model_name} - {problem_id}", 
                    color=colors[j], linestyle='--', marker=dotstyles[i], markersize=8)
            
    plt.xlabel("Parameter Combination")
    plt.ylabel("Mean Performance")
    plt.title("Performance Comparison")
    
    # 自定义图例：颜色代表ProblemID，点型代表Model
    color_handles = [Line2D([0], [0], color=colors[j], lw=2) for j in range(len(problem_ids))]
    dotstyle_handles = [Line2D([0], [0], color='black', marker=dotstyles[i], linestyle='None', markersize=8) for i in range(len(models))]

    legend1 = plt.legend(color_handles, [f'Problem {pid}' for pid in problem_ids], title='Problem ID', loc='upper left', bbox_to_anchor=(1.01, 1))
    legend2 = plt.legend(dotstyle_handles, models, title='Model', loc='upper left', bbox_to_anchor=(1.01, 0.6))
    plt.gca().add_artist(legend1)
    
    plt.show()

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    document_label = None 
    
    models = ["Qwen", "bert", "bge"]
    #models = ["Qwen"]
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
    #seeds = [860]

    param_comb = [["self"], ["self", "prompt"], ["self", "prompt", "peer"]]#[["self"], ["prompt"], ["peer"], ["self", "peer"], ["self", "prompt"], ["self", "peer", "prompt"]]

    performance_metric = "CORR" # "CORR" or "MSE"
    optimization_goal = 'CORR' # 'CORR' or 'MSE'
    example_num = 150
    
    performanceDict = {}
    dataset = pd.read_csv('Data/CPSTfulldataset2.csv', encoding="utf-8-sig")
    problem_ids = sorted(dataset["ProblemID"].unique())
    #problem_ids = ["Becky"]
    for model_name in models:
        print("Current Model = {}".format(model_name))
        performanceDict[model_name] = {}
        if document_label is None:
            file_path = script_dir / f'Results/{model_name}/'
        else:
            file_path = script_dir / f'Results/{document_label}/{model_name}/'
        full_dataset_path = file_path / 'final.csv'
        full_dataset = pd.read_csv(full_dataset_path, encoding="utf-8-sig")

        performance_values = []
        performance_std = []
        for problem in problem_ids:
            print("Processing {}".format(problem))
            show_statistic(full_dataset[full_dataset["ProblemID"] == problem])
            performanceDict[model_name][problem] = {}
            
            all_values = [[] for _ in range(len(param_comb))]
            for seed in seeds:
                training_data, testing_data = get_data(
                    full_dataset, problem_id=problem, example_num=example_num,
                    dsi_type="DSI", prompt_type="CosDis", peer_type="CosDis",
                    seed=seed, reallocate_testing_set=True
                )
                for idx, params in enumerate(param_comb):
                    corr, mse = calculate_performance(training_data, testing_data, parameters=params, optimization_goal=optimization_goal)
                    value = mse if performance_metric == "MSE" else corr
                    all_values[idx].append(value)
                    print(f"Seed {seed}, Params {params}: {performance_metric} = {value:.5f}")
                    print("\n")
                    
            # Save mean and std for each param_comb
            means = [np.mean(vals) for vals in all_values]
            stds = [np.std(vals) for vals in all_values]
            performance_values.append(means)
            performance_std.append(stds)
            
            for idx, params in enumerate(param_comb):
                performanceDict[model_name][problem][', '.join(params)] = {
                    'mean': means[idx],
                    'std': stds[idx]
                }
                
        performance_values = np.array(performance_values)
        performance_std = np.array(performance_std)
        
        draw(problem_ids, performance_values, performance_std, model_name, example_num, sample_num, performance_metric, optimization_goal, file_path, param_comb, save_pic=True)
