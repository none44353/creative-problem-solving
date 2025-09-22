import os
import numpy as np

from matplotlib import pyplot as plt


def plot_attacked_performance_comparison(all_problem_IDs, performance_values, performance_std, attack_method, attacked_performance_values, attacked_performance_std, model_name, example_num, sample_num, param_comb, performance_measure="spearman correlation", font_style="Tiro Bangla", save_fig=False, save_path=None):
    """
    Plot a comparison of performance values before and after an attack.

    Args:
        all_problem_IDs: a list of problem IDs.
        performance_values: a 2D numpy array of performance values (rows: problems, columns: parameter combinations).
        performance_std: a 2D numpy array of standard deviations for performance values.
        attack_method: the name of the attack method used.
        attacked_performance_values: a 2D numpy array of performance values after the attack.
        attacked_performance_std: a 2D numpy array of standard deviations for attacked performance values.
        model_name: the name of the model used.
        example_num: the number of examples used in training.
        sample_num: the number of train-test split samples.
        param_comb: a list of parameter combinations (each combination is a list of parameter codes).
        performance_measure: the measure used for the evaluation of the performance. Defaults to "spearman correlation".
        font_style: the font style of the plot. Defaults to "Tiro Bangla".
        save_fig: whether to save the plot. Defaults to False.
        save_path: the directory to save the plot. Defaults to True. Need to be specified when save_fig is set to True.
    """
    
    x = np.arange(len(all_problem_IDs))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Translate parameter codes to readable strings.
    str_trans = {"self": "DSI", "prompt": "PromptCosDis", "peer": "PeerCosDis"}
    labels = [' > '.join(str_trans[param] for param in params) for params in param_comb]
    
    # Set colors for bars and attacked bars.
    colors = ['#f8bdbd', '#bdf8bd', '#bdbdf8']
    attacked_colors = ['#c88d8d', '#8dc88d', '#8d8dc8']
    
    # Set font style as desired.
    plt.rcParams["font.family"] = font_style

    # Plot bars with error bars and annotations of unattacked performance.
    width = 0.2
    for idx, label in enumerate(labels):
        ax.bar(x + (idx - 1) * width, performance_values[:, idx], width, label=label, color=colors[idx])
        ax.errorbar(x + (idx - 1) * width, performance_values[:, idx], yerr=performance_std[:, idx], fmt='none', ecolor='#7f7f7f', capsize=3, elinewidth=1)
        for i in range(len(all_problem_IDs)):
            ax.text(x[i] + (idx - 1) * width, performance_values[i, idx]+.001, f"{performance_values[i, idx]:.3f}", ha='center', va='bottom', fontsize=9, c="#7f7f7f")
    
    # And do the same for attacked performance.   
    for idx, label in enumerate(labels):
        ax.bar(x + (idx - 1) * width, attacked_performance_values[:, idx], width, label=label + " (attacked)", color=attacked_colors[idx], hatch='//', alpha=.5)
        ax.errorbar(x + (idx - 1) * width, attacked_performance_values[:, idx], yerr=attacked_performance_std[:, idx], fmt='none', ecolor='black', capsize=3, elinewidth=1)
        for i in range(len(all_problem_IDs)):
            ax.text(x[i] + (idx - 1) * width, attacked_performance_values[i, idx]+.001, f"{attacked_performance_values[i, idx]:.3f}", ha='center', va='bottom', fontsize=9)

    # Set the legend, the label and the ticks of x & y axes, and the title of the plot.
    ax.legend(ncol=2, loc='upper left')
    ax.set_ylabel('Correlation', fontdict={'family': font_style})
    ax.set_title(f'{" ".join([word.capitalize() for word in performance_measure.split(" ")])} Comparison (Model: {model_name.upper()}, #Example: {example_num}, #Sample: {sample_num}, Attack Method: {attack_method})', fontdict={'family': font_style})
    ax.set_yticks(np.round(np.arange(0, .8, .1), 1))
    ax.set_yticklabels(ax.get_yticks(), fontdict={'family': font_style})
    ax.set_xticks(x)
    ax.set_xticklabels(all_problem_IDs, ha='center', fontdict={'family': font_style})
    plt.tight_layout()
    
    # Show or save the plot according to the specified parameter.
    if save_fig:
        assert save_path is not None, "Please specify a directory to save the figure when you want to (and hence set the param `save_fig` to True)."
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"Self-Prompt-Peer:{' '.join([word.capitalize() for word in performance_measure.split(' ')])}-Comparison-{example_num}-shot-{sample_num}-sample-{attack_method}-attacked.jpg"))
    else:
        plt.show()
    