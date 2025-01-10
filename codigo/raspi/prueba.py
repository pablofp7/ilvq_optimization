import csv

# Datos de ejemplo para un nodo
nodo = {
    "id": 0,
    "precision": 0.499,
    "recall": 0.573,
    "f1": 0.533,
    "muestras_train": 1000,
    "protos_train": 0,
    "shared_times_final": 0,
    "compartidos_final": 0,
    "no_comp_jsd_final": 0,
    "protos_descartados_final": 0,
    "tiempo_learn_data": 12.415695924311876,
    "clust_runs": 3,
    "clust_time": 0.7,
    "tiempo_learn_queue": 99.583438469,
    "tiempo_share_final": 0.007205231813713908,
    "tiempo_no_share_final": 110.43450664100237,
    "tiempo_espera_total": 100.35425060195848,
    "tiempo_final_total": 110.52080610697158,
    "cap_ejec": 8.929,
    "tam_lotes_recibidos": [],
    "tam_conj_prot": [(0, 0), (10, 6), (20, 13), (30, 20), (40, 27), (50, 34)]
}

# Crear una fila con los datos del nodo
row = {
    "NODO": nodo["id"],
    "Precision": nodo["precision"],
    "Recall": nodo["recall"],
    "F1": nodo["f1"],
    "Muestras entrenadas": nodo["muestras_train"],
    "Prototipos entrenados": nodo["protos_train"],
    "Veces compartido": nodo["shared_times_final"],
    "Prototipos compartidos": nodo["compartidos_final"],
    "Prototipos ahorrados": nodo["no_comp_jsd_final"],
    "Prototipos descartados": nodo["protos_descartados_final"],
    "Tiempo aprendizaje (muestras)": nodo["tiempo_learn_data"],
    "Ejecuciones de clustering": nodo["clust_runs"],
    "Tiempo clustering": nodo["clust_time"],
    "Tiempo aprendizaje (prototipos)": nodo["tiempo_learn_queue"],
    "Tiempo compartiendo prototipos": nodo["tiempo_share_final"],
    "Tiempo no compartiendo prototipos": nodo["tiempo_no_share_final"],
    "Tiempo total espera activa": nodo["tiempo_espera_total"],
    "Tiempo total": nodo["tiempo_final_total"],
    "Capacidad de ejecución": nodo["cap_ejec"],
    "Tamaño de lotes recibidos": nodo["tam_lotes_recibidos"],
    "Tamaño conjunto de prototipos": nodo["tam_conj_prot"]
}

# Nombre del archivo CSV
nombre_archivo = "resultados_simulacion.csv"

# Escribir en el archivo CSV
with open(nombre_archivo, mode='w', newline='') as csvfile:
    fieldnames = row.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Escribir la cabecera (nombres de las columnas)
    writer.writeheader()
    # Escribir la fila de datos
    writer.writerow(row)

print(f"Se ha generado el archivo CSV: {nombre_archivo}")