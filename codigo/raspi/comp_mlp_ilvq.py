import pandas as pd
import time
from river import metrics, tree, forest
from prototypes.xuilvq import XuILVQ
from mlp.mlp import DynamicMLP  # Import your class correctly

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
    beta = 0.5
    fbeta = round((1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall), 3) if (precision + recall) != 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "fbeta": fbeta}

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

@measure_time
def evaluate_model_sliding_window(model, dataset, window_size=100):
    """
    Evaluates a model using a sliding window of recent samples.

    Args:
        model: The model to evaluate.
        dataset (pd.DataFrame): The dataset for evaluation.
        window_size: The size of the sliding window.

    Returns:
        dict: A confusion matrix (TP, TN, FP, FN).
    """
    conf_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for _, row in dataset.iterrows():
        # Extract features and labels
        x = {i: row[col] for i, col in enumerate(feature_columns)}
        y = row[label_column]

        prediction = model.predict_one(x)
        model.learn_sliding_window(x, y, window_size=window_size)

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

@measure_time
def evaluate_model_batch_training(model, dataset, batch_size=32):
    """
    Evaluates a model using batch training.

    Args:
        model: The model to evaluate.
        dataset (pd.DataFrame): The dataset for evaluation.
        batch_size: The size of each batch.

    Returns:
        dict: A confusion matrix (TP, TN, FP, FN).
    """
    conf_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    batch_samples = []
    batch_labels = []

    for _, row in dataset.iterrows():
        # Extract features and labels
        x = {i: row[col] for i, col in enumerate(feature_columns)}
        y = row[label_column]

        # Add to batch
        batch_samples.append(x)
        batch_labels.append(y)

        # Train on batch if it's full
        if len(batch_samples) == batch_size:
            model.learn_batch(batch_samples, batch_labels)
            batch_samples = []
            batch_labels = []

        # Predict and update confusion matrix
        prediction = model.predict_one(x)
        if prediction == 0 and y == 0:
            conf_matrix["TN"] += 1
        elif prediction == 1 and y == 1:
            conf_matrix["TP"] += 1
        elif prediction == 1 and y == 0:
            conf_matrix["FP"] += 1
        else:
            conf_matrix["FN"] += 1

    # Train on the remaining samples in the last batch
    if batch_samples:
        model.learn_batch(batch_samples, batch_labels)

    return conf_matrix

# Dictionary of dataset names and file paths
data_name = {
    "elec": "electricity.csv",
}

# Load dataset
name = "elec"
dataset = read_dataset(name, data_name)
dataset = dataset.iloc[:10000]

# Separate features and labels
feature_columns = dataset.columns[:-1]  # All except the last column
label_column = dataset.columns[-1]      # The last column is the label

# Hyperparameters
learning_rate = 0.01
dropout_rate = 0.1
hidden_units = [32, 16]
weight_decay = 0.1
window_size = 32  
batch_size = 512   

# Initialize the MLP model
model_mlp = DynamicMLP(
    input_size=len(feature_columns),
    hidden_units=hidden_units,
    learning_rate=learning_rate,
    dropout_rate=dropout_rate,
    weight_decay=weight_decay
)

# Evaluate MLP in online learning mode
print("\nEvaluating MLP in online learning mode...")
conf_matrix_online, elapsed_time_online = evaluate_model_online_learning(model_mlp, dataset)
metrics_online = calculate_metrics(conf_matrix_online)
print(f"\nMetrics for MLP (Online Learning):")
for name, value in metrics_online.items():
    print(f"{name.capitalize()}: {value:.4f}")
print(f"Execution time: {elapsed_time_online:.2f} seconds")

# Evaluate MLP in sliding window mode
print("\nEvaluating MLP in sliding window mode...")
conf_matrix_sliding, elapsed_time_sliding = evaluate_model_sliding_window(model_mlp, dataset, window_size=window_size)
metrics_sliding = calculate_metrics(conf_matrix_sliding)
print(f"\nMetrics for MLP (Sliding Window):")
for name, value in metrics_sliding.items():
    print(f"{name.capitalize()}: {value:.4f}")
print(f"Execution time: {elapsed_time_sliding:.2f} seconds")

# Evaluate MLP in batch training mode
print("\nEvaluating MLP in batch training mode...")
conf_matrix_batch, elapsed_time_batch = evaluate_model_batch_training(model_mlp, dataset, batch_size=batch_size)
metrics_batch = calculate_metrics(conf_matrix_batch)
print(f"\nMetrics for MLP (Batch Training):")
for name, value in metrics_batch.items():
    print(f"{name.capitalize()}: {value:.4f}")
print(f"Execution time: {elapsed_time_batch:.2f} seconds")

# Initialize other models
model_xuilvq = XuILVQ()
model_hoeffding = tree.HoeffdingTreeClassifier()
model_adaptive_rf = forest.ARFClassifier()

models = {
    "XuILVQ": model_xuilvq,
    "Hoeffding Tree": model_hoeffding,
    "Adaptive Random Forest": model_adaptive_rf,
}

# Evaluate other models
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    conf_matrix, elapsed_time = evaluate_model_online_learning(model, dataset)
    metrics = calculate_metrics(conf_matrix)
    print(f"\nMetrics for {model_name}:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")
    print(f"Execution time: {elapsed_time:.2f} seconds")