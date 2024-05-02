import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import numpy as np
import pandas as pd     
from prototypes_mod import XuILVQ
from prototypes import XuILVQ
from sklearn.cluster import DBSCAN
from tqdm import tqdm




def read_dataset():
    dataset = pd.read_csv(f"../dataset/electricity.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset


def dbscan_prototypes(modelo, max_prototypes=100, target_range=(80, 90), eps_initial=0.0001):
    original_count = len(modelo.buffer.prototypes)
    if original_count <= max_prototypes:
        print("No need to run DBSCAN.")
        return
    
    original_prototypes = modelo.buffer.prototypes.copy()
    target_min, target_max = int(target_range[0] * original_count / 100), int(target_range[1] * original_count / 100)
    
    min_samples = 1
    iterations = 0
    previous_value = original_count
    eps = eps_initial
    last_eps = eps_initial
    
    ajuste_grueso = True
    lower_eps = eps_initial
    upper_eps = eps_initial

    while True:
        new_prototypes = {}
        next_prototype_id = 1

        # eps = float(input("Introduce un valor de eps:"))
        
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
                sum_m = sum(original_prototypes[i+1]['m'] for i in cluster_indices)

                new_prototypes[next_prototype_id] = {
                    'x': centroid,
                    'y': label,
                    'm': sum_m,
                    'neighbors': []
                }
                next_prototype_id += 1

        current_prototype_count = len(new_prototypes)
        # print(f"Prototypes after DBSCAN iteration {iterations}: {current_prototype_count}")

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
                eps = (lower_eps + upper_eps) / 2
                # print(f"Ajuste fino, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_max: {target_max}. Previous value: {previous_value}")
            elif current_prototype_count < target_min:
                lower_eps = eps
                eps = (lower_eps + upper_eps) / 2
                # print(f"Ajuste fino, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_min: {target_min}. Previous value: {previous_value}")
                
        
        if iterations > 100:
            print("Max iterations reached. Stopping.")
            break
        
        iterations += 1


    # print(f"Conjunto de prototipos antes: {modelo.buffer.prototypes}")
    modelo.buffer.prototypes = new_prototypes
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

        for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing with LIMIT={LIMIT}, Target={target_range}")):
            x = {k: v for k, v in enumerate(x)}

            if len(modelo.buffer.prototypes) > LIMIT:
                dbscan_prototypes(modelo, max_prototypes=LIMIT, target_range=target_range)
                rebuild_neighborhoods(modelo)

            prediction = modelo.predict_one(x)
            modelo.learn_one(x, y)

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

        precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
        recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        results.append({
            "LIMIT": LIMIT,
            "Target Range": target_range,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"Finished: LIMIT={LIMIT}, Target={target_range}, Precision={precision}, Recall={recall}, F1={f1}")

#After that, make the BASE experiment with the base model

df = read_dataset()
df_list = [(fila[:-1], fila[-1]) for fila in df.values]

modelo = XuILVQ(gamma=150)
matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing with LIMIT={LIMIT}, Target=ORIGINAL")):
    x = {k: v for k, v in enumerate(x)}

    prediction = modelo.predict_one(x)
    modelo.learn_one(x, y)

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

precision = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FP"]) if matriz_conf["TP"] + matriz_conf["FP"] > 0 else 0
recall = matriz_conf["TP"] / (matriz_conf["TP"] + matriz_conf["FN"]) if matriz_conf["TP"] + matriz_conf["FN"] > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

results.append({
    "LIMIT": "BASE",
    "Target Range": "ORIGINAL",
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})



# Optionally, convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

    



