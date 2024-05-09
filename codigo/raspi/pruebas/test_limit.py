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




def read_dataset():
    # dataset = pd.read_csv(f"../dataset/electricity.csv")
    dataset = pd.read_csv(f"../dataset/kdd99_http.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset[:100]



limit_values = [50, 100, 150, 250, 500]
fixed_ranges = [(80, 90), (70, 80), (60, 70), (50, 60)]
base_target = 75
widths = [5, 10, 15, 20, 25]
variable_ranges = [(max(base_target - w/2, 0), min(base_target + w/2, 100)) for w in widths]
target_ranges = fixed_ranges + variable_ranges

results = []

for LIMIT in limit_values:
    for target_range in target_ranges:
        df = read_dataset()
        df_list = [(fila[:-1], fila[-1]) for fila in df.values]

        modelo = MiILVQ(max_pset_size=LIMIT, target_size=target_range)
        matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        train_time = 0
        prediction_time = 0
        train_operations = 0
        prediction_operations = 0
        iteration_time = 0
        iteration_count = 0
        dbscan_count = 0
        rebuild_count = 0
        
        prev_eps = None
        aux_eps = None

        for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing with LIMIT={LIMIT}, Target={target_range}")):
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
                    
            print(f"Prediction: {prediction}")
            print(f"Y: {y}")
            
            if modelo.dbscan_count != prev_eps:
                dbscan_count += 1
                prev_eps = modelo.dbscan_count
                
            if modelo.rebuild_count != aux_eps:
                rebuild_count += 1
                aux_eps = modelo.rebuild_count
                    
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
            "LIMIT": LIMIT,
            "Target Range": target_range,
            "F1 Score": round(f1, 4),
            "Avg Time Iteration": avg_iteration_time,
            "Avg Time Train": avg_train_time,
            "Avg Time Prediction": avg_prediction_time
        })
        
        print(f"Finished: LIMIT={LIMIT}, Target={target_range}, Precision={precision}, Recall={recall}, F1={f1}, iteration time: {iteration_time / 1e9}")


#After that, make the BASE experiment with the base model

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
        prediction = prediction.get(1.0, 0.0)

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
        "LIMIT": "ORIGINAL",
        "Target Range": "BASE",
        "F1 Score": round(f1, 4),
        "Avg Time Iteration": avg_iteration_time,
        "Avg Time Train": avg_train_time,
        "Avg Time Prediction": avg_prediction_time
    })

results = sorted(results, key=lambda x: x['F1 Score'], reverse=True)

# Optionally, convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

with open("results_test_limit.txt", "w") as f:
    f.write(results_df.to_string())

    



