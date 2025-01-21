import time
import pandas as pd


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
    # Separate features and labels
    feature_columns = dataset.columns[:-1]  # All except the last column
    label_column = dataset.columns[-1]      # The last column is the label
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

def calculate_metrics(conf_matrix):
    """
    Calculate precision, recall, and F1-score from a confusion matrix in dictionary format.

    Args:
        conf_matrix: Confusion matrix in dictionary format {"TP", "TN", "FP", "FN"}.

    Returns:
        precision, recall, f1: Computed metrics.
    """
    TP = conf_matrix["TP"]
    TN = conf_matrix["TN"]
    FP = conf_matrix["FP"]
    FN = conf_matrix["FN"]

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
