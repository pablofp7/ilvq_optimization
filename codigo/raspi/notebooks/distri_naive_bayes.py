import pandas as pd
import time
from river import naive_bayes

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

def share_parameters(model):
    """
    Extracts the parameters of the Naive Bayes model to share with neighbors.

    Args:
        model: The Naive Bayes model (e.g., `river.naive_bayes.GaussianNB`).

    Returns:
        dict: A dictionary containing the model's parameters.
    """
    if not hasattr(model, "summary"):
        raise ValueError("The model does not have a summary attribute (not a GaussianNB model).")

    # Extract means and variances for each class
    parameters = {}
    for class_name, stats in model.summary.items():
        parameters[class_name] = {
            "mean": stats.mean,
            "variance": stats.variance,
            "n_samples": stats.n,
        }
    return parameters

def aggregate_parameters(model, neighbors_parameters):
    """
    Aggregates parameters received from neighbors into the local model.

    Args:
        model: The local Naive Bayes model.
        neighbors_parameters (list): A list of dictionaries containing parameters from neighbors.

    Returns:
        None: The local model is updated in-place.
    """
    if not hasattr(model, "summary"):
        raise ValueError("The model does not have a summary attribute (not a GaussianNB model).")

    for neighbor_params in neighbors_parameters:
        for class_name, stats in neighbor_params.items():
            if class_name not in model.summary:
                # If the class is not in the local model, initialize it
                model.summary[class_name] = model._new_stats()
                model.classes_.add(class_name)

            # Update the local model's statistics using weighted averaging
            local_stats = model.summary[class_name]
            neighbor_stats = stats

            # Combine means and variances using weighted averaging
            total_samples = local_stats.n + neighbor_stats["n_samples"]
            if total_samples > 0:
                local_stats.mean = (local_stats.mean * local_stats.n + neighbor_stats["mean"] * neighbor_stats["n_samples"]) / total_samples
                local_stats.variance = (local_stats.variance * local_stats.n + neighbor_stats["variance"] * neighbor_stats["n_samples"]) / total_samples
                local_stats.n = total_samples

# Dictionary of dataset names and file paths
data_name = {
    "elec": "electricity.csv",
}

# Load dataset
name = "elec"
dataset = read_dataset(name, data_name)
dataset = dataset.iloc[:5000]  # Use a subset for faster testing

# Separate features and labels
feature_columns = dataset.columns[:-1]  # All except the last column
label_column = dataset.columns[-1]      # The last column is the label

# Initialize models for each node
node_models = {
    "node_1": naive_bayes.GaussianNB(),
    "node_2": naive_bayes.GaussianNB(),
    "node_3": naive_bayes.GaussianNB(),
}

# Train each model on local data (simulated)
for node_name, model in node_models.items():
    print(f"Training {node_name}...")
    for _, row in dataset.iterrows():
        x = {i: row[col] for i, col in enumerate(feature_columns)}
        y = row[label_column]
        model.learn_one(x, y)

# Share parameters between nodes
node_parameters = {}
for node_name, model in node_models.items():
    node_parameters[node_name] = share_parameters(model)

# Simulate parameter sharing (e.g., node_1 receives parameters from node_2 and node_3)
print("\nAggregating parameters for node_1...")
neighbors_parameters = [node_parameters["node_2"], node_parameters["node_3"]]
aggregate_parameters(node_models["node_1"], neighbors_parameters)

# Evaluate the updated model on node_1
print("\nEvaluating node_1 after aggregation...")
conf_matrix, elapsed_time = evaluate_model_online_learning(node_models["node_1"], dataset)
metrics = calculate_metrics(conf_matrix)

# Print results
print("\nMetrics for node_1 after aggregation:")
for name, value in metrics.items():
    print(f"{name.capitalize()}: {value:.4f}")
print(f"Execution time: {elapsed_time:.2f} seconds")