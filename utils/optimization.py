import pygad
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.linear_model import Lasso, Ridge, LinearRegression

from utils.calculate_corr import calculate_correlation_after_outlier_removal, calculate_mse_after_outlier_removal


def get_tuple(data, dsi_type, prompt_type, peer_type):
    if dsi_type == "DSI":
        dsi_value = data.DSI
    else:
        raise ValueError("Unsupported dsi_type: {}".format(dsi_type))

    if prompt_type == "CosDis":
        prompt_value = data.promptCosDis
    else:
        raise ValueError("Unsupported prompt_type: {}".format(prompt_type))

    if peer_type == "CosDis":
        peer_value = data.peerCosDis
    else:
        raise ValueError("Unsupported peer_type: {}".format(peer_type))

    return (dsi_value, prompt_value, peer_value, data.FacScoresO)

#【改动】：将第一个参数由full_dataset修改为train_dataset
def ExampleGenerator(train_dataset, problem_id, example_num, dsi_type, prompt_type, peer_type, seed=0):
    assert isinstance(example_num, int) and example_num > 0, "example_num must be a positive integer"
    
    if problem_id != "Mike":
        example_items = train_dataset[train_dataset['ProblemID'] == problem_id].sample(n=example_num, random_state=seed, replace=False) #强调不放回采样，每个元素在采样中只会出现一次
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]
    elif problem_id == "Mike":
        selected_problem_id = train_dataset['ProblemID'].sample(n=1, random_state=seed).item()
        example_items = train_dataset[train_dataset['ProblemID'] == selected_problem_id].sample(n=example_num, random_state=seed, replace=False)
        # 修改：让Mike数据集也从自己的data里选
        # example_items = train_dataset[train_dataset['ProblemID'] == 'Mike'].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]

    return examples

def get_data(full_dataset, problem_id, example_num, dsi_type = "DSI", prompt_type = "CosDis", peer_type = "CosDis", seed = 0, reallocate_testing_set = False):
    if reallocate_testing_set:
        '''针对出了Mike以外的每个数据集，从中随机采样example_num个样本作为训练集'''
        train_dataset = pd.DataFrame()
        for pid in full_dataset['ProblemID'].unique():
            if pid != "Mike":
                pid_data = full_dataset[full_dataset['ProblemID'] == pid]
                train_samples = pid_data.sample(n=example_num, random_state=seed, replace=False)
                train_dataset = pd.concat([train_dataset, train_samples])
    else:
        train_dataset = full_dataset[full_dataset['set'] == 'training']

    examples = ExampleGenerator(train_dataset, problem_id, dsi_type=dsi_type, prompt_type=prompt_type, peer_type=peer_type, example_num=example_num, seed=seed)
    X_train = np.array([item[:-1] for item in examples])
    y_train = np.array([item[-1] for item in examples])

    test_items = full_dataset[(full_dataset['ProblemID'] == problem_id) & (~full_dataset.index.isin(train_dataset.index))]
    test_data = [get_tuple(row, dsi_type, prompt_type, peer_type) for row in test_items.itertuples()]
    X_test = np.array([item[:-1] for item in test_data])
    y_test = np.array([item[-1] for item in test_data])
    
    return (X_train, y_train), (X_test, y_test)

def get_key_parameters(X, key_parameters):
    Y = []
    for item in X:
        new_tuple = []
        if "self" in key_parameters:
            new_tuple.append(item[0])
        if "prompt" in key_parameters:
            new_tuple.append(item[1])
        if "peer" in key_parameters:
            new_tuple.append(item[2])
        Y.append(tuple(new_tuple))
    return Y
     
# # 用进化算法直接优化预测模型在training data上的correlation
# def fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes):
#     y_pred = np.dot(X_train, solution[:-1]) + solution[-1]

#     cutoff = 4 / (len(y_train) - 2) if len(y_train) > 2 else 0.1
#     correlation_value = calculate_correlation_after_outlier_removal(y_train, y_pred, cutoff)
#     return correlation_value

# # 假设这是用于运行进化算法的函数
# def genetic_algorithm(X_train, y_train, num_genes):
#     fitness_function = lambda ga_instance, solution, solution_idx: fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes)

#     ga_instance = pygad.GA(
#         num_generations=200,
#         num_parents_mating=20,
#         fitness_func=fitness_function,
#         sol_per_pop=40,
#         num_genes=num_genes,  # 对应 lambda1, lambda2, lambda3, lambda4
#         parent_selection_type="rws", #rws: 适应度越高的个体被选中的概率越大，sss: 选择适应度最高的个体
#         crossover_type="two_points",
#         mutation_type="random",
#         mutation_percent_genes="default"
#     )

#     ga_instance.run()
#     best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
#     return best_solution
    

def optimize_linear_corr(X_train, y_train, method, alpha):
    '''根据X_train和y_train训练一个线性回归模型，使用Scipy的自定义优化函数，目标是最大化预测值与真实值y_train之间的pearsonr相关系数'''
    def objective(params):
        weights = params[:-1]
        intercept = params[-1]
        y_pred = np.dot(X_train, weights) + intercept
        cutoff = 4 / (len(y_train) - 2) if len(y_train) > 2 else 0.1
        corr = calculate_correlation_after_outlier_removal(y_train, y_pred, cutoff)
        
        # 目标是最大化相关系数，因此返回负值
        if method == "Linear":
            target = -corr
        elif method == "Lasso":
            target = -corr + alpha * np.sum(np.abs(params))
        elif method == "Ridge":
            target = -corr + alpha * np.sum(params**2)
        
        return target

    best_corr = -np.inf
    best_solution = None
    initial_params_list = [
        np.concatenate(([1], np.zeros(X_train.shape[1]))),
        np.concatenate(([-1], np.zeros(X_train.shape[1]))),
    ]
    # for _ in range(8):
    #     rand_params = np.random.uniform(-1, 1, X_train.shape[1] + 1)
    #     while np.allclose(rand_params, 0):
    #         rand_params = np.random.uniform(-1, 1, X_train.shape[1] + 1)
    #     initial_params_list.append(rand_params)
        
    for initial_params in initial_params_list:
        result = minimize(objective, initial_params, method='BFGS')
        current_corr = -result.fun
        if current_corr > best_corr:
            best_corr = current_corr
            best_solution = result.x
    return best_solution

    initial_params = np.zeros(X_train.shape[1] + 1)
    initial_params[0] = 1
    result = minimize(objective, initial_params, method='BFGS')
    
    return result.x

def calculate_performance(training_data, testing_data, parameters, optimization_goal, method="Linear", alpha=None, max_iter=10000):
    full_X_train, y_train = training_data
    full_X_test, y_test = testing_data
    X_train = get_key_parameters(full_X_train, parameters)
    X_test = get_key_parameters(full_X_test, parameters)

    if optimization_goal == 'MSE':
        if method == "Linear":
            model = LinearRegression(fit_intercept=True)
        elif method == "Lasso":
            model = Lasso(alpha=alpha if alpha is not None else 0.1, fit_intercept=True, max_iter=max_iter)
        elif method == "Ridge":
            model = Ridge(alpha=alpha if alpha is not None else 1.0, fit_intercept=True, max_iter=max_iter)
        else:
            raise ValueError("Unsupported method: {}".format(method))
            
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print("Model coefficients & intercept:", model.coef_, model.intercept_)

        prediction_train = model.predict(X_train)
        training_corr = calculate_correlation_after_outlier_removal(y_train, prediction_train)
        training_mse = calculate_mse_after_outlier_removal(y_train, prediction_train)
        print("Training Performance: MSE={}, CORR={}".format(training_mse, training_corr))
    
    elif optimization_goal == 'CORR':
        # print("Training Data:", [data[0].item() for data in X_train[:5]], [data.item() for data in y_train[:5]])
        solution = optimize_linear_corr(np.array(X_train), np.array(y_train), method=method, alpha=alpha)
        prediction = np.dot(np.array(X_test), solution[:-1]) + solution[-1]
        print("Model coefficients & intercept:", solution)
        
        prediction_train = np.dot(np.array(X_train), solution[:-1]) + solution[-1]
        training_corr = calculate_correlation_after_outlier_removal(y_train, prediction_train)
        training_mse = calculate_mse_after_outlier_removal(y_train, prediction_train)
        print("Training Performance: MSE={}, CORR={}".format(training_mse, training_corr))
        
        # solution = genetic_algorithm(X_train, y_train, num_genes= len(parameters) + 1)
        # prediction = np.dot(X_test, solution[:-1]) + solution[-1]
        # print("Model coefficients & intercept:", solution)
        
    prediction_scaled = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)
    #print("Predictions (first 5):", prediction_scaled[:5])

    corr = calculate_correlation_after_outlier_removal(y_test, prediction)
    mse = calculate_mse_after_outlier_removal(y_test, prediction)
    return corr, mse, training_corr, training_mse
