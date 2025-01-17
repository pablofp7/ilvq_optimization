import pandas as pd
from river import metrics, tree, forest
from prototypes_mod.xuilvq import XuILVQ

# Cargar el dataset generado en CSV
csv_path = "airlines_normalized.csv"
df = pd.read_csv(csv_path)

# Reducir el dataset a las primeras 50k filas, si aplica
df = df.head(50000)

# Preparar los datos para River
dataset = [({col: row[col] for col in df.columns[:-1]}, row["Delay"]) for _, row in df.iterrows()]

# Inicializar los modelos
model_xuilvq = XuILVQ()
model_hoeffding = tree.HoeffdingTreeClassifier()
model_adaptive_rf = forest.ARFClassifier()

# Inicializar las métricas
metrics_xuilvq = {
    "accuracy": metrics.Accuracy(),
    "precision": metrics.Precision(),
    "recall": metrics.Recall(),
    "f1": metrics.F1()
}

metrics_hoeffding = {
    "accuracy": metrics.Accuracy(),
    "precision": metrics.Precision(),
    "recall": metrics.Recall(),
    "f1": metrics.F1()
}

metrics_adaptive_rf = {
    "accuracy": metrics.Accuracy(),
    "precision": metrics.Precision(),
    "recall": metrics.Recall(),
    "f1": metrics.F1()
}

# Evaluar el modelo XuILVQ
print("Evaluando XuILVQ...")
for x, y in dataset:
    y_pred = model_xuilvq.predict_one(x)  # Predicción
    model_xuilvq.learn_one(x, y)          # Entrenamiento incremental
    for metric in metrics_xuilvq.values():
        metric.update(y, y_pred)

# Mostrar métricas para XuILVQ
print("\nMétricas para XuILVQ:")
for name, metric in metrics_xuilvq.items():
    print(f"{name.capitalize()}: {metric.get():.4f}")

# Evaluar el modelo Hoeffding Tree
print("\nEvaluando Hoeffding Tree...")
for x, y in dataset:
    y_pred = model_hoeffding.predict_one(x)  # Predicción
    model_hoeffding.learn_one(x, y)          # Entrenamiento incremental
    for metric in metrics_hoeffding.values():
        metric.update(y, y_pred)

# Mostrar métricas para Hoeffding Tree
print("\nMétricas para Hoeffding Tree:")
for name, metric in metrics_hoeffding.items():
    print(f"{name.capitalize()}: {metric.get():.4f}")

# Evaluar el modelo Adaptive Random Forest
print("\nEvaluando Adaptive Random Forest...")
for x, y in dataset:
    y_pred = model_adaptive_rf.predict_one(x)  # Predicción
    model_adaptive_rf.learn_one(x, y)          # Entrenamiento incremental
    for metric in metrics_adaptive_rf.values():
        metric.update(y, y_pred)

# Mostrar métricas para Adaptive Random Forest
print("\nMétricas para Adaptive Random Forest:")
for name, metric in metrics_adaptive_rf.items():
    print(f"{name.capitalize()}: {metric.get():.4f}")

# Comparación final
print("\nComparación de resultados:")
for metric_name in metrics_xuilvq.keys():
    print(f"{metric_name.capitalize()} - XuILVQ: {metrics_xuilvq[metric_name].get():.4f}, Hoeffding Tree: {metrics_hoeffding[metric_name].get():.4f}, Adaptive RF: {metrics_adaptive_rf[metric_name].get():.4f}")
