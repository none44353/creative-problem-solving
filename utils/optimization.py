from utils.calculate_corr import calculate_correlation_after_outlier_removal
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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

def ExampleGenerator(full_dataset, problem_id, example_num, dsi_type, prompt_type, peer_type, seed=0):
    assert isinstance(example_num, int) and example_num > 0, "example_num must be a positive integer"
    train_dataset = full_dataset[full_dataset['set'] == 'training']

    if problem_id != "Mike":
        example_items = train_dataset[train_dataset['ProblemID'] == problem_id].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]
    elif problem_id == "Mike":
        selected_problem_id = train_dataset['ProblemID'].sample(n=1, random_state=seed).item()
        example_items = train_dataset[train_dataset['ProblemID'] == selected_problem_id].sample(n=example_num, random_state=seed)
        # 修改：让Mike数据集也从自己的data里选
        # example_items = full_dataset[full_dataset['ProblemID'] == 'Mike'].sample(n=example_num, random_state=seed)
        examples = [get_tuple(example, dsi_type, prompt_type, peer_type) for example in example_items.itertuples()]

    return examples

def get_data(full_dataset, problem_id, example_num, dsi_type = "DSI", prompt_type = "CosDis", peer_type = "CosDis", seed = 0):
    examples = ExampleGenerator(full_dataset, problem_id, dsi_type=dsi_type, prompt_type=prompt_type, peer_type=peer_type, example_num=example_num, seed=seed)
    X_train = np.array([item[:-1] for item in examples])
    y_train = np.array([item[-1] for item in examples])

    test_items = full_dataset[(full_dataset['ProblemID'] == problem_id) & (full_dataset['set'] != 'training')]
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
     
# 用进化算法直接优化预测模型在training data上的correlation
def fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes):
    y_pred = np.dot(X_train, solution[:-1]) + solution[-1]

    cutoff = 4 / (len(y_train) - 2) if len(y_train) > 2 else 0.1
    correlation_value = calculate_correlation_after_outlier_removal(y_train, y_pred, cutoff)
    return correlation_value

# 假设这是用于运行进化算法的函数
def genetic_algorithm(X_train, y_train, num_genes):
    fitness_function = lambda ga_instance, solution, solution_idx: fitness_function_template(ga_instance, solution, solution_idx, X_train, y_train, num_genes)

    ga_instance = pygad.GA(
        num_generations=200,
        num_parents_mating=20,
        fitness_func=fitness_function,
        sol_per_pop=40,
        num_genes=num_genes,  # 对应 lambda1, lambda2, lambda3, lambda4
        parent_selection_type="rws", #rws: 适应度越高的个体被选中的概率越大，sss: 选择适应度最高的个体
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes="default"
    )

    ga_instance.run()
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
    return best_solution
    
def calculate_performance(training_data, testing_data, parameters, optimization_goal):
    full_X_train, y_train = training_data
    full_X_test, y_test = testing_data
    X_train = get_key_parameters(full_X_train, parameters)
    X_test = get_key_parameters(full_X_test, parameters)

    if optimization_goal == 'MSE':
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print("Model coefficients & intercept:", model.coef_, model.intercept_)
    elif optimization_goal == 'correlation':
        solution = genetic_algorithm(X_train, y_train, num_genes= len(parameters) + 1)
        prediction = np.dot(X_test, solution[:-1]) + solution[-1]
        print("Model coefficients & intercept:", solution)
        
    prediction_scaled = 4 * (prediction - prediction.min()) / (prediction.max() - prediction.min()) if prediction.max() > prediction.min() else np.zeros_like(prediction)
    # print("Predictions (first 5):", prediction_scaled[:5])
        
    cutoff = 4 / (len(y_test) - 2) if len(y_test) > 2 else 0.1
    corr, y_test_clean, pred_clean = calculate_correlation_after_outlier_removal(y_test, prediction, cutoff)
    mse = mean_squared_error(y_test_clean, pred_clean)
    return corr, mse