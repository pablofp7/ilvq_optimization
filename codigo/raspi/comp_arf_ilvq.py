import pandas as pd
import time
from river import metrics, tree, forest, linear_model, naive_bayes
from prototypes.xuilvq import XuILVQ
from prototypes_mod.xuilvq import XuILVQ as UpdatedILVQ

# Decorator to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        return result, elapsed_time
    return wrapper

def read_dataset(name: str, data_name: dict):
    """
    Reads and preprocesses the dataset.

    Args:
        name (str): The name of the dataset to load.
        data_name (dict): A dictionary mapping dataset names to file paths.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    filename = data_name[name]
    dataset = pd.read_csv(f"./dataset/{filename}")
    # Replace categorical values with numeric ones
    dataset.replace({'UP': 1, 'DOWN': 0, 'True': 1, 'False': 0}, inplace=True)
    dataset.infer_objects(copy=False)
    return dataset

# Custom function to calculate metrics from confusion matrix
def calculate_metrics(conf_matrix):
    """
    Calculates precision, recall, and F1-score from a confusion matrix.

    Args:
        conf_matrix (dict): A dictionary with keys TP, TN, FP, FN.

    Returns:
        dict: A dictionary with precision, recall, and F1-score.
    """
    tp = conf_matrix["TP"]
    fp = conf_matrix["FP"]
    fn = conf_matrix["FN"]

    precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0
    recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if precision + recall != 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

@measure_time
def evaluate_model_online_learning(model, dataset):
    """
    Evaluates a model in online learning mode (one sample at a time).

    Args:
        model: The model to evaluate.
        dataset (pd.DataFrame): The dataset for evaluation.

    Returns:
        dict: A confusion matrix (TP, TN, FP, FN).
    """
    conf_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for _, row in dataset.iterrows():
        # Extract features and labels
        x = {i: row[col] for i, col in enumerate(feature_columns)}
        y = row[label_column]

        prediction = model.predict_one(x)
        model.learn_one(x, y)

        # Update confusion matrix
        if prediction == 0 and y == 0:
            conf_matrix["TN"] += 1
        elif prediction == 1 and y == 1:
            conf_matrix["TP"] += 1
        elif prediction == 1 and y == 0:
            conf_matrix["FP"] += 1
        else:
            conf_matrix["FN"] += 1

    return conf_matrix

# Dictionary of dataset names and file paths
data_name = {
    "elec": "electricity.csv",
}

# Load dataset
name = "elec"
dataset = read_dataset(name, data_name)
dataset = dataset.iloc[:5000]

# Separate features and labels
feature_columns = dataset.columns[:-1]  # All except the last column
label_column = dataset.columns[-1]      # The last column is the label

# Define parameter grid for ARF
n_models_list = [1, 2, 3, 5, 10]  
n_models_list = []
max_size_list = [5, 10, 20, 50]  

# Initialize models
models = {
    "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
    "ILVQ": XuILVQ(),
    "Updated ILVQ": UpdatedILVQ(),
    "Online Logistic Regression": linear_model.LogisticRegression(),  
    "Online Naive Bayes": naive_bayes.GaussianNB(),  
}

# Add ARF with different parameter combinations
for n_models in n_models_list:
    for max_size in max_size_list:
        models[f"ARF (n_models={n_models}, max_size={max_size})"] = forest.ARFClassifier(
            n_models=n_models,
            max_size=max_size
        )

# Evaluate models
results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    conf_matrix, elapsed_time = evaluate_model_online_learning(model, dataset)
    metrics = calculate_metrics(conf_matrix)
    results[model_name] = {
        "metrics": metrics,
        "time": elapsed_time,
    }
    print(f"\nMetrics for {model_name}:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")
    print(f"Execution time: {elapsed_time:.2f} seconds")

# Separate Hoeffding Tree and ILVQ results
hoeffding_result = {"Hoeffding Tree": results["Hoeffding Tree"]}
ilvq_result = {"ILVQ": results["ILVQ"]}
updated_ilvq_result = {"Updated ILVQ": results["Updated ILVQ"]}

# Separate ARF results
arf_results = {k: v for k, v in results.items() if k.startswith("ARF")}

# Sort ARF results by F1 score (descending)
sorted_arf_results = dict(sorted(arf_results.items(), key=lambda x: x[1]["metrics"]["f1"], reverse=True))

# Combine all results
final_results = {**hoeffding_result, **ilvq_result, **sorted_arf_results}

# Display results
print("\nComparison of results across models (sorted by F1 for ARF):")
for model_name, result in final_results.items():
    print(f"\n{model_name}:")
    for metric_name, value in result["metrics"].items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")
    print(f"  Execution time: {result['time']:.2f} seconds")