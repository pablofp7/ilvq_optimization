import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import numpy as np
import pandas as pd     
from prototypes_mod import XuILVQ
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import time




def read_dataset():
    dataset = pd.read_csv(f"../dataset/electricity.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset


def dbscan_prototypes(modelo, max_prototypes=100, target_range=(80, 90), eps_initial=0.000001):
    original_count = len(modelo.buffer.prototypes)
    if original_count <= max_prototypes:
        print("No need to run DBSCAN.")
        return
    
    original_prototypes = modelo.buffer.prototypes.copy()
    target_min, target_max = int(target_range[0] * original_count / 100), int(target_range[1] * original_count / 100)
    
    min_samples = 1
    iterations = 0
    previous_value = original_count
    ajuste_grueso = True
    
    if eps_initial is None:
        eps_initial = 0.000001

    eps = eps_initial
    last_eps = eps_initial
    lower_eps = eps_initial
    upper_eps = eps_initial
        


    while True:
        new_prototypes = {}
        next_prototype_id = 1

        for label in set(proto['y'] for proto in original_prototypes.values()):
            label_prototypes = np.array([proto['x'] for proto in original_prototypes.values() if proto['y'] == label])
            if label_prototypes.size == 0:
                continue

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(label_prototypes)

            for cluster_id in np.unique(labels):
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_protos = [label_prototypes[i] for i in cluster_indices]
                centroid = np.mean(cluster_protos, axis=0)

                sum_m = sum(original_prototypes[list(original_prototypes.keys())[i]]['m'] for i in cluster_indices)

                new_prototypes[next_prototype_id] = {
                    'x': centroid,
                    'y': label,
                    'm': sum_m,
                    'neighbors': []
                }
                next_prototype_id += 1

        current_prototype_count = len(new_prototypes)
        # print(f"Prototypes after DBSCAN iteration {iterations}: {current_prototype_count}. Objective: {target_min} - {target_max}")
        
        if target_min <= current_prototype_count <= target_max:
            # print(f"Prototypes within target range after {iterations} iterations. {current_prototype_count} prototypes.")
            break
        
        if ajuste_grueso:
            if current_prototype_count > target_max:
                if previous_value > target_max:
                    last_eps = eps
                    eps *= 10
                    # print(f"Ajuste grueso, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_max: {target_max}. Previous value: {previous_value}")
                else:
                    ajuste_grueso = False
                    upper_eps = eps
                    lower_eps = last_eps

                previous_value = current_prototype_count

            elif current_prototype_count < target_min:
                if previous_value < target_min:
                    last_eps = eps
                    eps /= 10
                    # print(f"Ajuste grueso, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_min: {target_min}. Previous value: {previous_value}")
                else:
                    ajuste_grueso = False
                    lower_eps = eps
                    upper_eps = last_eps
        
                previous_value = current_prototype_count
        
        else:
                
            if current_prototype_count > target_max:
                upper_eps = eps
            elif current_prototype_count < target_min:
                lower_eps = eps
            
            new_eps = (lower_eps + upper_eps) / 2
            if new_eps == eps:
                ajuste_grueso = True
            eps = new_eps
                
        
        if iterations > 100:
            print("Max iterations reached. Stopping.")
            with open("max_it_reached.txt", "w") as f:
                f.write("Max iterations reached. Stopping.")
            exit()
        
        iterations += 1

    # print(f"Conjunto de prototipos antes: {modelo.buffer.prototypes}")
    modelo.buffer.prototypes = new_prototypes
    return eps 
    # print(f"Conjunto de prototipos despues: {modelo.buffer.prototypes}")
    
    
def rebuild_neighborhoods(model, num_neighbors=2):
    buffer = model.buffer  
    
    proto_ids = list(buffer.prototypes.keys())
    proto_vectors = np.array([buffer.prototypes[pid]['x'] for pid in proto_ids])

    new_edges = {}

    for i, pid1 in enumerate(proto_ids):
        distances = [(pid2, buffer.get_distance(proto_vectors[i], proto_vectors[j])) 
                     for j, pid2 in enumerate(proto_ids) if pid1 != pid2]

        distances.sort(key=lambda x: x[1])
        closest_neighbors = [nid for nid, _ in distances[:num_neighbors]]

        # Update neighbor list for the current prototype
        buffer.prototypes[pid1]['neighbors'] = closest_neighbors
        for nid in closest_neighbors:
            new_edges[(pid1, nid)] = 1  

    # print(f"EDGES BEFORE: {buffer.edges_whole}")
    buffer.edges = new_edges
    # print(f"EDGES AFTER: {buffer.edges_whole}")


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
        # df_list = df_list[:20000]  # Using a subset for quicker runs

        modelo = XuILVQ()
        matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        train_time = 0
        prediction_time = 0
        train_operations = 0
        prediction_operations = 0
        total_dbscan_time = 0
        total_rebuild_time = 0
        iteration_time = 0
        iteration_count = 0
        dbscan_count = 0
        rebuild_count = 0
        
        prev_eps = None
        aux_eps = None

        for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing with LIMIT={LIMIT}, Target={target_range}")):
            start_iteration_time = time.perf_counter_ns()
            
            x = {k: v for k, v in enumerate(x)}

            if len(modelo.buffer.prototypes) > LIMIT:
                start_dbscan = time.perf_counter_ns()
                aux_eps = dbscan_prototypes(modelo, max_prototypes=LIMIT, target_range=target_range, eps_initial=prev_eps)
                prev_eps = aux_eps
                fin_dbscan = time.perf_counter_ns() - start_dbscan
                total_dbscan_time += fin_dbscan
                dbscan_count += 1
                 
                start_rebuild = time.perf_counter_ns()
                rebuild_neighborhoods(modelo)
                fin_rebuild = time.perf_counter_ns() - start_rebuild
                total_rebuild_time += fin_rebuild
                rebuild_count += 1

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
                    
            iteration_time += time.perf_counter_ns() - start_iteration_time
            iteration_count += 1

        precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
        recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        avg_train_time = train_time / train_operations if train_operations else 0
        avg_prediction_time = prediction_time / prediction_operations if prediction_operations else 0
        avg_dbscan_time = total_dbscan_time / dbscan_count if dbscan_count else 0
        avg_rebuild_time = total_rebuild_time / rebuild_count if rebuild_count else 0
        avg_iteration_time = iteration_time / iteration_count
        
        avg_train_time = avg_train_time / 1e9
        avg_prediction_time = avg_prediction_time / 1e9
        avg_dbscan_time = avg_dbscan_time / 1e9
        avg_rebuild_time = avg_rebuild_time / 1e9
        avg_iteration_time = avg_iteration_time / 1e9
        
        results.append({
            "LIMIT": LIMIT,
            "Target Range": target_range,
            "F1 Score": round(f1, 4),
            "Avg Time Iteration": avg_iteration_time,
            "Avg Time Train": avg_train_time,
            "Avg Time Prediction": avg_prediction_time,
            "Avg Time DBSCAN": avg_dbscan_time,
            "Avg Time Rebuild": avg_rebuild_time
        })
        
        print(f"Finished: LIMIT={LIMIT}, Target={target_range}, Precision={precision}, Recall={recall}, F1={f1}, iteration time: {iteration_time / 1e9}")


#After that, make the BASE experiment with the base model

df = read_dataset()
df_list = [(fila[:-1], fila[-1]) for fila in df.values]

modelo = XuILVQ(gamma=150)
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
avg_dbscan_time = 0
avg_rebuild_time = 0
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
        "Avg Time Prediction": avg_prediction_time,
        "Avg Time DBSCAN": avg_dbscan_time,
        "Avg Time Rebuild": avg_rebuild_time
    })

results = sorted(results, key=lambda x: x['F1 Score'], reverse=True)

# Optionally, convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

with open("results_test_limit.txt", "w") as f:
    f.write(results_df.to_string())

    



