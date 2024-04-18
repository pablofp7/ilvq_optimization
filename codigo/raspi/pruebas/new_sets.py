import sys
import os
import pandas as pd

# Configurando la ruta del directorio principal para asegurar el acceso a los módulos necesarios
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

# Importando módulos adicionales
import numpy as np
from prototypes import XuILVQ  # Asumo que este módulo es parte de tus archivos personalizados
import pickle
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import time

def preprocess_dataset(option: int):
    
    http = None
    movie = None
    
    if option == 0 or option == 2: 
        
        http = pd.read_csv("../dataset/kdd99_http.csv", sep=",")
        http.dropna(inplace=True)

    if option == 1 or option == 2:
        movie = pd.read_csv("../dataset/ml_100k.csv", sep='\t')
        movie['title'] = movie['title'].str.replace(r" \(\d{4}\)", "", regex=True)
        movie.dropna(inplace=True)
        movie.drop_duplicates(inplace=True)


        # Ensure all indices are reset before concatenation
        movie.reset_index(drop=True, inplace=True)

        # One-hot encode genres
        genres_expanded = movie['genres'].str.get_dummies(sep=',')
        genres_expanded.reset_index(drop=True, inplace=True)

        # Apply CountVectorizer to 'title'
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(movie['title'])
        titles_vec = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        titles_vec.reset_index(drop=True, inplace=True)

        # One-hot encode 'gender' and 'occupation' if they exist in the DataFrame
        gender_dummies = pd.get_dummies(movie['gender'], prefix='gender') if 'gender' in movie.columns else pd.DataFrame()
        occupation_dummies = pd.get_dummies(movie['occupation'], prefix='occupation') if 'occupation' in movie.columns else pd.DataFrame()
        
        # Concatenate all results
        movie = pd.concat([movie.drop(['genres', 'title', 'gender', 'occupation'], axis=1, errors='ignore'), genres_expanded, titles_vec, gender_dummies, occupation_dummies], axis=1)
        
        # Ensure the 'rating' column is the last one
        columns = [col for col in movie.columns if col != 'rating']
        columns.append('rating')
        movie = movie[columns]
        
        movie['zip_code'] = pd.to_numeric(movie['zip_code'], errors = 'coerce')

        movie.dropna(inplace=True)
        movie.drop_duplicates(inplace=True)

    if option == 0 or option == 2:        
        http.to_csv("../dataset/http_proc.csv")
    if option == 1 or option == 2:
        movie.to_csv("../dataset/movie_proc.csv")


    
def read_dataset():
    http, movie = None, None
    
    http = pd.read_csv("../dataset/http_proc.csv", low_memory=False)
    movie = pd.read_csv("../dataset/movie_proc.csv", low_memory=False)
    return http, movie


def get_metrics(matrix: dict):
    # Determine if the classification is binary or multi-class
    classes = matrix.get('classes')
    
    if classes is None:
        # Assuming binary classification if no 'classes' key is found
        classes = [0, 1]
    
    if len(classes) == 2:
        # Binary classification metrics
        try:
            precision = matrix["TP"] / (matrix["TP"] + matrix["FP"])
        except ZeroDivisionError:
            precision = 0

        try:
            recall = matrix["TP"] / (matrix["TP"] + matrix["FN"])
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        try:
            accuracy = (matrix["TP"] + matrix["TN"]) / sum(matrix.values())
        except ZeroDivisionError:
            accuracy = 0
        
    else:
        # Multi-class classification metrics
        precision_list = []
        recall_list = []
        f1_list = []

        for class_label in classes:
            TP = matrix.get(f'TP_{class_label}', 0)
            FP = matrix.get(f'FP_{class_label}', 0)
            FN = matrix.get(f'FN_{class_label}', 0)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Macro-average metrics
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_list) / len(f1_list)

        # Micro-average metrics
        TP_micro = sum(matrix.get(f'TP_{cls}', 0) for cls in classes)
        FP_micro = sum(matrix.get(f'FP_{cls}', 0) for cls in classes)
        FN_micro = sum(matrix.get(f'FN_{cls}', 0) for cls in classes)
        precision_micro = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) > 0 else 0
        recall_micro = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        
        # Combine macro and micro metrics
        precision = (precision, precision_micro)
        recall = (recall, recall_micro)
        f1 = (f1, f1_micro)
        # Accuracy is not directly meaningful in multiclass micro-average context
        accuracy = None

    return precision, recall, f1, accuracy



if __name__ == "__main__":

    # preprocess_dataset(2)

    predict_times_movie = []
    learn_times_movie = []
    predict_times_http = []
    learn_times_http = []

    # Cargar los datasets
    http, movie = read_dataset()

    if movie is not None:
        modelo_movie = XuILVQ()
        movie_list = [(fila[:-1], fila[-1]) for fila in movie.values]
        matrix_movie = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        lista_tam_conj_movie = []

        for i, movie in enumerate(movie_list):
            
            if i == 20:
                break

            protos = list(modelo_movie.buffer.prototypes.values())
            tam_num = len(protos)
            lista_tam_conj_movie.append((i, tam_num))

            print(f"Procesando película {i + 1} de {len(movie_list)}") if i % 1000 == 0 else None
            x, y = movie
            x = {k: v for k, v in enumerate(x)}
            
            # Medir tiempo de predict_one
            start_time = time.perf_counter()
            prediccion = modelo_movie.predict_one(x)
            end_time = time.perf_counter()
            predict_times_movie.append(end_time - start_time)
            
            print(f"Predicción: {prediccion}, Real: {y}")

            if isinstance(prediccion, dict):
                if 1.0 in prediccion:
                    prediccion = prediccion[1.0]
                else:
                    prediccion = 0.0      

            if prediccion == 0 and y == 0:
                matrix_movie["TN"] += 1
            elif prediccion == 1 and y == 1:
                matrix_movie["TP"] += 1
            elif prediccion == 1 and y == 0:
                matrix_movie["FP"] += 1
            elif prediccion == 0 and y == 1:
                matrix_movie["FN"] += 1
            
            # Medir tiempo de learn_one
            start_time = time.perf_counter()
            modelo_movie.learn_one(x, y)
            end_time = time.perf_counter()
            learn_times_movie.append(end_time - start_time)

    if http is not None:
        modelo_http = XuILVQ()
        http_list = [(fila[:-1], fila[-1]) for fila in http.values]
        matrix_http = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        lista_tam_conj_http = []

        for i, http in enumerate(http_list):
            
            if i == 20:
                break

            protos = list(modelo_http.buffer.prototypes.values())
            tam_num = len(protos)
            lista_tam_conj_http.append((i, tam_num))

            print(f"Procesando petición HTTP {i + 1} de {len(http_list)}") if i % 1000 == 0 else None
            x, y = http
            x = {k: v for k, v in enumerate(x)}

            # Medir tiempo de predict_one
            start_time = time.perf_counter()
            prediccion = modelo_http.predict_one(x)
            end_time = time.perf_counter()
            predict_times_http.append(end_time - start_time)
            
            print(f"Predicción: {prediccion}, Real: {y}")


            if isinstance(prediccion, dict):
                if 1.0 in prediccion:
                    prediccion = prediccion[1.0]
                else:
                    prediccion = 0.0      

            if prediccion == 0 and y == 0:
                matrix_http["TN"] += 1
                print(f"Esto ha sido TN, Predicción: {prediccion}, Real: {y}")
                
            elif prediccion == 1 and y == 1:
                matrix_http["TP"] += 1
                print(f"Esto ha sido TP, Predicción: {prediccion}, Real: {y}")
            elif prediccion == 1 and y == 0:
                matrix_http["FP"] += 1
                print(f"Esto ha sido FP, Predicción: {prediccion}, Real: {y}")
            elif prediccion == 0 and y == 1:
                matrix_http["FN"] += 1
                print(f"Esto ha sido FN, Predicción: {prediccion}, Real: {y}")
            
            # Medir tiempo de learn_one
            start_time = time.perf_counter()
            modelo_http.learn_one(x, y)
            end_time = time.perf_counter()
            learn_times_http.append(end_time - start_time)

    # Calcular y mostrar tiempos promedio
    avg_predict_time_movie = sum(predict_times_movie) / len(predict_times_movie) if predict_times_movie else 0
    avg_learn_time_movie = sum(learn_times_movie) / len(learn_times_movie) if learn_times_movie else 0
    avg_predict_time_http = sum(predict_times_http) / len(predict_times_http) if predict_times_http else 0
    avg_learn_time_http = sum(learn_times_http) / len(learn_times_http) if learn_times_http else 0

    print(f"Average Prediction Time for Movie Model: {avg_predict_time_movie:.5f} seconds")
    print(f"Average Learning Time for Movie Model: {avg_learn_time_movie:.5f} seconds")
    print(f"Average Prediction Time for HTTP Model: {avg_predict_time_http:.5f} seconds")
    print(f"Average Learning Time for HTTP Model: {avg_learn_time_http:.5f} seconds")
    
    
    
    metrics_movie = get_metrics(matrix_movie)
    metrics_http = get_metrics(matrix_http)
        
        
    # Calcular y mostrar métricas para HTTP
    precision_http, recall_http, f1_http, accuracy_http = get_metrics(metrics_http)
    print(f"HTTP Metrics: Precision={precision_http}, Recall={recall_http}, F1={f1_http}, Accuracy={accuracy_http}")

    # Calcular y mostrar métricas para películas
    precision_movie, recall_movie, f1_movie, accuracy_movie = get_metrics(metrics_movie)
    print(f"Movie Metrics: Precision={precision_movie}, Recall={recall_movie}, F1={f1_movie}, Accuracy={accuracy_movie}")


    # Asegúrate de que todas las listas tengan la misma longitud
    max_length = max(len(lista_tam_conj_http), len(lista_tam_conj_movie), len(predict_times_movie), len(predict_times_http))

    # Extender las listas más cortas con NaN para igualar la longitud
    def extend_list(lst, length):
        return lst + [np.nan] * (length - len(lst))

    lista_tam_conj_http_extended = extend_list(lista_tam_conj_http, max_length)
    lista_tam_conj_movie_extended = extend_list(lista_tam_conj_movie, max_length)
    learning_times_http_extended = extend_list(learn_times_http, max_length)
    learning_times_movie_extended = extend_list(learn_times_movie, max_length)
    predict_times_movie_extended = extend_list(predict_times_movie, max_length)
    predict_times_http_extended = extend_list(predict_times_http, max_length)

    # Crear un DataFrame con estos datos
    data = {
        "Size of Prototypes HTTP": lista_tam_conj_http_extended,
        "Size of Prototypes Movies": lista_tam_conj_movie_extended,
        "Learning Times Movies": learning_times_movie_extended,
        "Learning Times HTTP": learning_times_http_extended,
        "Prediction Times Movies": predict_times_movie_extended,
        "Prediction Times HTTP": predict_times_http_extended
    }
    df = pd.DataFrame(data)

    # Apply rolling mean to smooth the 'Times' columns
    df['Smoothed Learning Times Movies'] = df['Learning Times Movies'].rolling(window=3, min_periods=1).mean() 
    df['Smoothed Learning Times HTTP'] = df['Learning Times HTTP'].rolling(window=3, min_periods=1).mean()
    df['Smoothed Prediction Times Movies'] = df['Prediction Times Movies'].rolling(window=3, min_periods=1).mean()
    df['Smoothed Prediction Times HTTP'] = df['Prediction Times HTTP'].rolling(window=3, min_periods=1).mean()

    # Unpack the list of tuples for HTTP
    x_http, y_http = zip(*lista_tam_conj_http)

    # Unpack the list of tuples for Movies
    x_movies, y_movies = zip(*lista_tam_conj_movie)

    # Creating the figure and axes for the subplots
    plt.figure(figsize=(12, 15))  # Increased figure size to accommodate more subplots

    # First row: Evolution of the prototype set sizes
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(x_http, y_http, color='b')
    ax1.set_title('Evolución del Tamaño del Conjunto de Prototipos HTTP')
    ax1.set_xlabel('Índice de Muestra')
    ax1.set_ylabel('Tamaño del Conjunto de Prototipos')
    ax1.grid(True)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(x_movies, y_movies, color='r')
    ax2.set_title('Evolución del Tamaño del Conjunto de Prototipos Películas')
    ax2.set_xlabel('Índice de Muestra')
    ax2.set_ylabel('Tamaño del Conjunto de Prototipos')
    ax2.grid(True)

    # Second row: Smoothed prediction times
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(df['Smoothed Prediction Times Movies'], label='Smoothed Prediction Times (Movies)')
    ax3.set_title('Smoothed Movie Model Prediction Performance')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(df['Smoothed Prediction Times HTTP'], label='Smoothed Prediction Times (HTTP)')
    ax4.set_title('Smoothed HTTP Model Prediction Performance')
    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()

    # Third row: Smoothed learning times
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(df['Smoothed Learning Times Movies'], label='Smoothed Learning Times (Movies)', linestyle='--')
    ax5.set_title('Smoothed Movie Model Learning Performance')
    ax5.set_xlabel('Sample Number')
    ax5.set_ylabel('Time (seconds)')
    ax5.legend()

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(df['Smoothed Learning Times HTTP'], label='Smoothed Learning Times (HTTP)', linestyle='--')
    ax6.set_title('Smoothed HTTP Model Learning Performance')
    ax6.set_xlabel('Sample Number')
    ax6.set_ylabel('Time (seconds)')
    ax6.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Adjust horizontal spacing if needed 
    plt.show()
    
    
    # Printear los valores de las lista de tam de conjs
    print(lista_tam_conj_http)
    print(lista_tam_conj_movie)
    