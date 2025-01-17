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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from river.forest import ARFClassifier
from sklearn.dummy import DummyClassifier



def preprocess_dataset():
    
    http = None
    movie = None
    
        
    http = pd.read_csv("../dataset/kdd99_http.csv", sep=",")
    http.dropna(inplace=True)

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

    http.to_csv("../dataset/http_proc.csv")
    movie.to_csv("../dataset/movie_proc.csv")


    
def read_dataset():
    pd.set_option('future.no_silent_downcasting', True)
    http, movie = None, None
    
    http = pd.read_csv("../dataset/http_proc.csv", low_memory=False)
    movie = pd.read_csv("../dataset/movie_proc.csv", low_memory=False)
    return http, movie


def get_metrics(matrix: dict):
    # Determine if the classification is binary or multi-class
    classes = matrix.get('Number of Classes')
    
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

    # preprocess_dataset()

    # Cargar los datasets
    http, movie = read_dataset()
    

    predict_times_movie = []
    learn_times_movie = []
    predict_times_http = []
    learn_times_http = []


    if movie is not None:
        modelo_movie = XuILVQ()
        forest_movie = ARFClassifier()
        dummy_movie = DummyClassifier(strategy='stratified')
        movie_list = [(fila[:-1], fila[-1]) for fila in movie.values]
        predictions_movie = []
        true_labels_movie = []
        predictions_movie_forest = []
        true_labels_movie_forest = []
        predictions_movie_dummy = []
        true_labels_movie_dummy = []
        lista_tam_conj_movie = []

        # Primero se evalua el dummy
        features = [fila[:-1] for fila in movie_list]
        labels = [fila[-1] for fila in movie_list]
        dummy_movie.fit(features, labels)
        predictions_movie_dummy.extend(list(dummy_movie.predict(features)))
        true_labels_movie_dummy.extend(labels)

        
        for i, (x, y) in enumerate(movie_list):
            if i % 1001 == 1000:  # Assuming you only want to process the first 20 for testing purposes
                print(f"Muestra {i} de {len(movie_list)}. MOVIE")    
                # break
            
            protos = list(modelo_movie.buffer.prototypes.values())
            tam = len(protos)
            lista_tam_conj_movie.append((i, tam))
            x = {k: v for k, v in enumerate(x)}

            # Measure prediction time
            start_time = time.perf_counter()
            prediction = modelo_movie.predict_one(x)
            end_time = time.perf_counter()
            predict_times_movie.append(end_time - start_time)

            if isinstance(prediction, dict):
                if 1.0 in prediction:
                    prediction = prediction[1.0]
                else:
                    prediction = 0.0
            
            if prediction is None:
                prediction = 0.0    
                    
            predictions_movie.append(prediction)
            true_labels_movie.append(y)
            

            # Measure learning time
            start_time = time.perf_counter()
            modelo_movie.learn_one(x, y)
            end_time = time.perf_counter()
            learn_times_movie.append(end_time - start_time)

            pred = forest_movie.predict_one(x)
            if pred is not None:
                predictions_movie_forest.append(forest_movie.predict_one(x))
                true_labels_movie_forest.append(y)
            forest_movie.learn_one(x, y)

        # Calculate confusion matrix and metrics for movie model
        cm_movie = confusion_matrix(true_labels_movie, predictions_movie)
        print(f"Confusion matrix movie: {cm_movie}")
        print(f"True Positives: {cm_movie[1][1]}")
        print(f"True Negatives: {cm_movie[0][0]}")
        print(f"False Positives: {cm_movie[0][1]}")
        print(f"False Negatives: {cm_movie[1][0]}")
        precision_movie = precision_score(true_labels_movie, predictions_movie, average='weighted', zero_division=0)
        recall_movie = recall_score(true_labels_movie, predictions_movie, average='weighted', zero_division=0)
        f1_movie = f1_score(true_labels_movie, predictions_movie, average='weighted', zero_division=0)
        accuracy_movie = accuracy_score(true_labels_movie, predictions_movie)
        
        cm_movie_forest = confusion_matrix(true_labels_movie_forest, predictions_movie_forest)
        print(f"Confusion matrix movie FOREST: {cm_movie_forest}")
        print(f"True Positives: {cm_movie_forest[1][1]}")
        print(f"True Negatives: {cm_movie_forest[0][0]}")
        print(f"False Positives: {cm_movie_forest[0][1]}")
        print(f"False Negatives: {cm_movie_forest[1][0]}")
        precision_movie_forest = precision_score(true_labels_movie_forest, predictions_movie_forest, average='weighted', zero_division=0)
        recall_movie_forest = recall_score(true_labels_movie_forest, predictions_movie_forest, average='weighted', zero_division=0)
        f1_movie_forest = f1_score(true_labels_movie_forest, predictions_movie_forest, average='weighted', zero_division=0)
        accuracy_movie_forest = accuracy_score(true_labels_movie_forest, predictions_movie_forest)
        
        cm_movie_dummy = confusion_matrix(true_labels_movie_dummy, predictions_movie_dummy)
        print(f"Confusion matrix movie DUMMY: {cm_movie_dummy}")
        print(f"True Positives: {cm_movie_dummy[1][1]}")
        print(f"True Negatives: {cm_movie_dummy[0][0]}")
        print(f"False Positives: {cm_movie_dummy[0][1]}")
        print(f"False Negatives: {cm_movie_dummy[1][0]}")
        precision_movie_dummy = precision_score(true_labels_movie_dummy, predictions_movie_dummy, average='weighted', zero_division=0)
        recall_movie_dummy = recall_score(true_labels_movie_dummy, predictions_movie_dummy, average='weighted', zero_division=0)
        f1_movie_dummy = f1_score(true_labels_movie_dummy, predictions_movie_dummy, average='weighted', zero_division=0)
        accuracy_movie_dummy = accuracy_score(true_labels_movie_dummy, predictions_movie_dummy)

    if http is not None:
        modelo_http = XuILVQ()
        forest_http = ARFClassifier()
        dummy_http = DummyClassifier(strategy='stratified')
        http_list = [(fila[:-1], fila[-1]) for fila in http.values]
        predictions_http = []
        true_labels_http = []
        predictions_http_forest = []
        true_labels_http_forest = []
        predictions_http_dummy = []
        true_labels_http_dummy = []
        lista_tam_conj_http = []
        
        # Primero se evalua el dummy
        features = [fila[:-1] for fila in http_list]
        labels = [fila[-1] for fila in http_list]
        dummy_http.fit(features, labels)
        predictions_http_dummy.extend(list(dummy_http.predict(features)))
        true_labels_http_dummy.extend(labels)


        for i, (x, y) in enumerate(http_list):
            if i % 1001 == 1000:  # Similarly for HTTP model
                print(f"Muestra {i} de {len(http_list)}. HTTP")
                # break
            
            protos = list(modelo_http.buffer.prototypes.values())
            tam = len(protos)
            lista_tam_conj_http.append((i, tam))

            x = {k: v for k, v in enumerate(x)}

            # Measure prediction time
            start_time = time.perf_counter()
            prediction = modelo_http.predict_one(x)
            end_time = time.perf_counter()
            predict_times_http.append(end_time - start_time)
            
            if isinstance(prediction, dict):
                if 1.0 in prediction:
                    prediction = prediction[1.0]
                else:
                    prediction = 0.0
            
            if prediction is None:
                prediction = 0.0   

            # Adjust prediction format if necessary
            prediction = max(prediction.items(), key=lambda item: item[1])[0] if isinstance(prediction, dict) else prediction

            predictions_http.append(prediction)
            true_labels_http.append(y)

            # Measure learning time
            start_time = time.perf_counter()
            modelo_http.learn_one(x, y)
            end_time = time.perf_counter()
            learn_times_http.append(end_time - start_time)
            
            
            pred = forest_http.predict_one(x)
            if pred is not None:
                predictions_http_forest.append(forest_http.predict_one(x))
                true_labels_http_forest.append(y)
            forest_http.learn_one(x, y)            


        # Calculate confusion matrix and metrics for HTTP model
        cm_http = confusion_matrix(true_labels_http, predictions_http)
        print(f"Confusion matrix http: {cm_http}")
        print(f"True Positives: {cm_http[1][1]}")
        print(f"True Negatives: {cm_http[0][0]}")
        print(f"False Positives: {cm_http[0][1]}")  
        print(f"False Negatives: {cm_http[1][0]}")
        precision_http = precision_score(true_labels_http, predictions_http, average='weighted', zero_division=0)
        recall_http = recall_score(true_labels_http, predictions_http, average='weighted', zero_division=0)
        f1_http = f1_score(true_labels_http, predictions_http, average='weighted', zero_division=0)
        accuracy_http = accuracy_score(true_labels_http, predictions_http)
        
        cm_http_forest = confusion_matrix(true_labels_http_forest, predictions_http_forest)
        print(f"Confusion matrix http FOREST: {cm_http_forest}")
        print(f"True Positives: {cm_http_forest[1][1]}")
        print(f"True Negatives: {cm_http_forest[0][0]}")
        print(f"False Positives: {cm_http_forest[0][1]}")
        print(f"False Negatives: {cm_http_forest[1][0]}")   
        precision_http_forest = precision_score(true_labels_http_forest, predictions_http_forest, average='weighted', zero_division=0)
        recall_http_forest = recall_score(true_labels_http_forest, predictions_http_forest, average='weighted', zero_division=0)
        f1_http_forest = f1_score(true_labels_http_forest, predictions_http_forest, average='weighted', zero_division=0)
        accuracy_http_forest = accuracy_score(true_labels_http_forest, predictions_http_forest)
        
        cm_http_dummy = confusion_matrix(true_labels_http_dummy, predictions_http_dummy)
        print(f"Confusion matrix http DUMMY: {cm_http_dummy}")
        print(f"True Positives: {cm_http_dummy[1][1]}")
        print(f"True Negatives: {cm_http_dummy[0][0]}")
        print(f"False Positives: {cm_http_dummy[0][1]}")
        print(f"False Negatives: {cm_http_dummy[1][0]}")
        precision_http_dummy = precision_score(true_labels_http_dummy, predictions_http_dummy, average='weighted', zero_division=0)
        recall_http_dummy = recall_score(true_labels_http_dummy, predictions_http_dummy, average='weighted', zero_division=0)
        f1_http_dummy = f1_score(true_labels_http_dummy, predictions_http_dummy, average='weighted', zero_division=0)
        accuracy_http_dummy = accuracy_score(true_labels_http_dummy, predictions_http_dummy)

    # Output the metrics and times
    print(f"Average Prediction Time for Movie Model: {sum(predict_times_movie) / len(predict_times_movie):.5f} seconds")
    print(f"Average Learning Time for Movie Model: {sum(learn_times_movie) / len(learn_times_movie):.5f} seconds")
    print(f"Average Prediction Time for HTTP Model: {sum(predict_times_http) / len(predict_times_http):.5f} seconds")
    print(f"Average Learning Time for HTTP Model: {sum(learn_times_http) / len(learn_times_http):.5f} seconds")
    print(f"Movie Metrics: Precision={precision_movie}, Recall={recall_movie}, F1={f1_movie}, Accuracy={accuracy_movie}")
    print(f"Movie (Forest) Metrics: Precision={precision_movie_forest}, Recall={recall_movie_forest}, F1={f1_movie_forest}, Accuracy={accuracy_movie_forest}")
    print(f"Movie (Dummy) Metrics: Precision={precision_movie_dummy}, Recall={recall_movie_dummy}, F1={f1_movie_dummy}, Accuracy={accuracy_movie_dummy}")
    print(f"HTTP Metrics: Precision={precision_http}, Recall={recall_http}, F1={f1_http}, Accuracy={accuracy_http}")
    print(f"HTTP (Forest) Metrics: Precision={precision_http_forest}, Recall={recall_http_forest}, F1={f1_http_forest}, Accuracy={accuracy_http_forest}")
    print(f"HTTP (Dummy) Metrics: Precision={precision_http_dummy}, Recall={recall_http_dummy}, F1={f1_http_dummy}, Accuracy={accuracy_http_dummy}")
    


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
    df['Smoothed Learning Times Movies'] = df['Learning Times Movies'].rolling(window=10, min_periods=1).mean() 
    df['Smoothed Learning Times HTTP'] = df['Learning Times HTTP'].rolling(window=10, min_periods=1).mean()
    df['Smoothed Prediction Times Movies'] = df['Prediction Times Movies'].rolling(window=10, min_periods=1).mean()
    df['Smoothed Prediction Times HTTP'] = df['Prediction Times HTTP'].rolling(window=10, min_periods=1).mean()

    # Unpack the list of tuples for HTTP
    x_http, y_http = zip(*lista_tam_conj_http)

    # Unpack the list of tuples for Movies
    x_movies, y_movies = zip(*lista_tam_conj_movie)


    # Plot for the first row: Evolution of the prototype set sizes
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(x_http, y_http, color='b')
    ax1.set_title('Evolution of Protoype Set Size. HTTP')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Prototype Set Size')
    ax1.grid(True)

    ax2.plot(x_movies, y_movies, color='r')
    ax2.set_title('Evolution of Protoype Set Size. Movies')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Prototype Set Size')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("grafica_newset_size.png")

    # Plot for the second row: Smoothed prediction times
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
    ax3.plot(df['Smoothed Prediction Times Movies'], label='Smoothed Prediction Times (Movies)')
    ax3.set_title('Smoothed Movie Model Prediction Performance')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()

    ax4.plot(df['Smoothed Prediction Times HTTP'], label='Smoothed Prediction Times (HTTP)')
    ax4.set_title('Smoothed HTTP Model Prediction Performance')
    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig("grafica_newset_prediction.png")

    # Plot for the third row: Smoothed learning times
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 5))
    ax5.plot(df['Smoothed Learning Times Movies'], label='Smoothed Learning Times (Movies)', linestyle='--')
    ax5.set_title('Smoothed Movie Model Learning Timing')
    ax5.set_xlabel('Sample Number')
    ax5.set_ylabel('Time (seconds)')
    ax5.legend()

    ax6.plot(df['Smoothed Learning Times HTTP'], label='Smoothed Learning Times (HTTP)', linestyle='--')
    ax6.set_title('Smoothed HTTP Model Learning Timing')
    ax6.set_xlabel('Sample Number')
    ax6.set_ylabel('Time (seconds)')
    ax6.legend()

    plt.tight_layout()
    plt.savefig("grafica_newset_training.png")
        # plt.show()
    
    