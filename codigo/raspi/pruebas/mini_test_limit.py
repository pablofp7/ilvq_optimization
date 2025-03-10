import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import numpy as np
import pandas as pd     
from prototypes_mod import XuILVQ as MiILVQ
from prototypes import XuILVQ
from tqdm import tqdm
import time

# Function to read the dataset
def read_dataset():
    pd.set_option('future.no_silent_downcasting', True)
    dataset = pd.read_csv(f"../dataset/electricity.csv")
    dataset.replace('UP', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('DOWN', 0, inplace=True) 
    dataset.infer_objects(copy=False)
    dataset.replace('True', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('False', 0, inplace=True) 
    dataset.infer_objects(copy=False)
    return dataset

# Initialize results list
results = []

# DBSCAN configurations
dbscan_configs = [
    (50, (50, 60)),
    (150, (50, 60)),
    (250, (50, 60)),
    (500, (72.5, 77.5))
]

# Iterate over DBSCAN configurations
for LIMIT, target_range in dbscan_configs:
    df = read_dataset()
    df_list = [(fila[:-1], fila[-1]) for fila in df.values]

    modelo = MiILVQ(max_pset_size=LIMIT, target_size=target_range, merge_mode="dbscan")
    matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    train_time = 0
    prediction_time = 0
    train_operations = 0
    prediction_operations = 0
    iteration_time = 0
    iteration_count = 0

    for i, (x, y) in enumerate(tqdm(df_list, desc=f"DBSCAN: LIMIT={LIMIT}, Target={target_range}")):
        start_iteration_time = time.perf_counter_ns()
        x = {k: v for k, v in enumerate(x)}

        start_time = time.perf_counter_ns()
        prediction = modelo.predict_one(x)
        prediction_time += time.perf_counter_ns() - start_time
        prediction_operations += 1

        start_time = time.perf_counter_ns()
        modelo.learn_one(x, y)
        train_time += time.perf_counter_ns() - start_time
        train_operations += 1

        if isinstance(prediction, dict):
            if 1.0 in prediction:
                prediction = prediction[1.0]
            else:
                prediction = 0.0  

        if prediction is not None and prediction == y:
            if prediction == 1.0:
                matriz_conf["TP"] += 1
            else:
                matriz_conf["TN"] += 1
        elif prediction is not None:
            if prediction == 1.0:
                matriz_conf["FP"] += 1
            else:
                matriz_conf["FN"] += 1

        iteration_time += time.perf_counter_ns() - start_iteration_time
        iteration_count += 1

    precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
    recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    avg_train_time = train_time / train_operations if train_operations else 0
    avg_prediction_time = prediction_time / prediction_operations if prediction_operations else 0
    avg_iteration_time = iteration_time / iteration_count

    avg_train_time = avg_train_time / 1e9
    avg_prediction_time = avg_prediction_time / 1e9
    avg_iteration_time = avg_iteration_time / 1e9

    results.append({
        "Method": "DBSCAN",
        "Limit Size": LIMIT,
        "Target Range": target_range,
        "F1 Score": round(f1, 4),
        "Avg Time Iteration": avg_iteration_time,
        "Avg Time Train": avg_train_time,
        "Avg Time Prediction": avg_prediction_time
    })

    print(f"Finished: DBSCAN: LIMIT={LIMIT}, Target={target_range}, Precision={precision}, Recall={recall}, F1={f1}, iteration time: {iteration_time / 1e9}")
    print(f"Matriz de confusion: {matriz_conf}")

# K-Means configurations
kmeans_configs = [
    (50, 50), (50, 90),
    (100, 50), (100, 90),
    (250, 50), (250, 90),
    (500, 50), (500, 90),
    (1000, 50), (1000, 90)
]

# Add 75% goal to all combinations
kmeans_configs_with_75 = kmeans_configs + [(limit, 75) for limit, _ in kmeans_configs]

# Iterate over K-Means configurations
for limit_size, goal_percentage in kmeans_configs_with_75:
    df = read_dataset()
    df_list = [(fila[:-1], fila[-1]) for fila in df.values]

    modelo = MiILVQ(max_pset_size=limit_size, target_percentage=goal_percentage, merge_mode="kmeans")
    matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    train_time = 0
    prediction_time = 0
    train_operations = 0
    prediction_operations = 0
    iteration_time = 0
    iteration_count = 0

    for i, (x, y) in enumerate(tqdm(df_list, desc=f"K-Means: Limit Size={limit_size}, Goal Percentage={goal_percentage}%")):
        start_iteration_time = time.perf_counter_ns()
        x = {k: v for k, v in enumerate(x)}

        start_time = time.perf_counter_ns()
        prediction = modelo.predict_one(x)
        prediction_time += time.perf_counter_ns() - start_time
        prediction_operations += 1

        start_time = time.perf_counter_ns()
        modelo.learn_one(x, y)
        train_time += time.perf_counter_ns() - start_time
        train_operations += 1

        if isinstance(prediction, dict):
            if 1.0 in prediction:
                prediction = prediction[1.0]
            else:
                prediction = 0.0  

        if prediction is not None and prediction == y:
            if prediction == 1.0:
                matriz_conf["TP"] += 1
            else:
                matriz_conf["TN"] += 1
        elif prediction is not None:
            if prediction == 1.0:
                matriz_conf["FP"] += 1
            else:
                matriz_conf["FN"] += 1

        iteration_time += time.perf_counter_ns() - start_iteration_time
        iteration_count += 1

    precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
    recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    avg_train_time = train_time / train_operations if train_operations else 0
    avg_prediction_time = prediction_time / prediction_operations if prediction_operations else 0
    avg_iteration_time = iteration_time / iteration_count

    avg_train_time = avg_train_time / 1e9
    avg_prediction_time = avg_prediction_time / 1e9
    avg_iteration_time = avg_iteration_time / 1e9

    results.append({
        "Method": "K-Means",
        "Limit Size": limit_size,
        "Target Range": goal_percentage,
        "F1 Score": round(f1, 4),
        "Avg Time Iteration": avg_iteration_time,
        "Avg Time Train": avg_train_time,
        "Avg Time Prediction": avg_prediction_time
    })

    print(f"Finished: K-Means: Limit Size={limit_size}, Goal Percentage={goal_percentage}%, Precision={precision}, Recall={recall}, F1={f1}, iteration time: {iteration_time / 1e9}")
    print(f"Matriz de confusion: {matriz_conf}")

# Process the base model
df = read_dataset()
df_list = [(fila[:-1], fila[-1]) for fila in df.values]

modelo = XuILVQ()
matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
train_time = 0
prediction_time = 0
iteration_time = 0
train_operations = 0
prediction_operations = 0
count_iterations = 0

for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing with BASE, Target=ORIGINAL")):
    start_iteration_time = time.perf_counter_ns()
    x = {k: v for k, v in enumerate(x)}

    start_time = time.perf_counter_ns()
    prediction = modelo.predict_one(x)
    prediction_time += time.perf_counter_ns() - start_time
    prediction_operations += 1

    start_time = time.perf_counter_ns()
    modelo.learn_one(x, y)
    train_time += time.perf_counter_ns() - start_time
    train_operations += 1

    if isinstance(prediction, dict):
        if 1.0 in prediction:
            prediction = prediction[1.0]
        else:
            prediction = 0.0  

    if prediction is not None and prediction == y:
        if prediction == 1.0:
            matriz_conf["TP"] += 1
        else:
            matriz_conf["TN"] += 1
    elif prediction is not None:
        if prediction == 1.0:
            matriz_conf["FP"] += 1
        else:
            matriz_conf["FN"] += 1
    
    count_iterations += 1
    iteration_time += time.perf_counter_ns() - start_iteration_time

precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

avg_train_time = train_time / train_operations if train_operations else 0
avg_prediction_time = prediction_time / prediction_operations if prediction_operations else 0
avg_iteration_time = iteration_time / count_iterations

avg_train_time = avg_train_time / 1e9
avg_prediction_time = avg_prediction_time / 1e9
avg_iteration_time = avg_iteration_time / 1e9

results.append({
    "Method": "BASE",
    "Limit Size": "ORIGINAL",
    "Target Range": "BASE",
    "F1 Score": round(f1, 4),
    "Avg Time Iteration": avg_iteration_time,
    "Avg Time Train": avg_train_time,
    "Avg Time Prediction": avg_prediction_time
})

# Sort and print the results
results = sorted(results, key=lambda x: x['F1 Score'], reverse=True)
results_df = pd.DataFrame(results)
print(results_df)

# Save the results to a file
with open("mini_results_test_limit.txt", "w") as f:
    f.write(results_df.to_string())
