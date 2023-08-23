from ..IA.prototypes import XuILVQ
import numpy as np
import pandas as pd



def read_dataset():
    
    dataset = pd.read_csv('../IA/electricity.csv')
    # Se descarta la primera fila del dataset que contiene los nombres de las columnas
    # dataset = dataset.drop(0, axis=0)

    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 
              
    return dataset

def update_matrix(prediccion, y):
    
    global matriz_confusion, to_write
    
    if prediccion == 0 and y == 0:
        matriz_confusion["TN"] += 1
    elif prediccion == 1 and y == 1:
        matriz_confusion["TP"] += 1
    elif prediccion == 1 and y == 0:
        matriz_confusion["FP"] += 1
    else:
        matriz_confusion["FN"] += 1
        
    try:
        PRECISION = matriz_confusion["TP"] / (matriz_confusion["TP"] + matriz_confusion["FP"])
        RECALL = matriz_confusion["TP"] / (matriz_confusion["TP"] + matriz_confusion["FN"])
        F1 = 2 * ((PRECISION * RECALL) / (PRECISION + RECALL))
    except ZeroDivisionError:
        PRECISION = 0
        RECALL = 0
        F1 = 0
    
    str1 = f"\n\nMatriz de confusión: {matriz_confusion}. Número de muestras: {sum(matriz_confusion.values())}" 
    str2 = f"\nPrecision: {PRECISION}"
    str3 = f"\nRecall: {RECALL}"
    str4 = f"\nF1: {F1}\n\n"
    
    print(str1)
    print(str2)
    print(str3)
    print(str4)
    
    to_write.append(str1)
    to_write.append(str2)
    to_write.append(str3)
    to_write.append(str4)    
    
    return PRECISION, RECALL, F1


def write_to_file(to_write):
    
    nombre_archivo = 'salida.txt'

    # Abrir el archivo en modo escritura
    with open(nombre_archivo, 'w') as archivo:
        # Escribir cada string en una línea separada
        for string in to_write:
            archivo.write(string)

    # Cerrar el archivo
    archivo.close()


def entropia(conj_proto: list):
    
    #Convertir la lista de listas a arrays de numpy
    array_proto = np.array(conj_proto)
    
    #Transformar el conjunto de arrays de numpy en un array de numpy
    flat_array = array_proto.flatten()
    
    # print(f"flat_array: {flat_array}")
    
    unique_values, value_counts = np.unique(flat_array, return_counts=True)
    prob_values = value_counts / len(flat_array)
    entropy = -np.sum(prob_values * np.log2(prob_values))
    
    return entropy

def get_prototypes(model):
    
    # print(f"Se printean los prototipos generados por el modelo:\n")
    conjunto_prototipos = []
    prototypes = model.buffer.prototypes.copy()
    for proto in prototypes.values():
        
        # try:
        #     del proto['neighbors']
        # except KeyError:
        #     pass
        
        # try:
        #     del proto['m']
        # except KeyError:
        #     pass
        
        vec = proto['x'].tolist()
        vec.append(proto['y'])
        conjunto_prototipos.append(vec)
        # print(f"{vec}")
    
    return conjunto_prototipos
    
def main():
    dataset = read_dataset()
    model = XuILVQ()
    model2 = XuILVQ()
    model3 = XuILVQ()
    model4 = XuILVQ()
    global matriz_confusion 
    
    train_samples = 200


    for i, muestra in zip(range(train_samples), dataset.values):
        print(f"MUESTRA: {i}")
        prediccion = model.predict_one({k: v for k, v in enumerate(muestra[:-1])})
        update_matrix(prediccion, muestra[-1])
             
        model.learn_one({k: v for k, v in enumerate(muestra[:-1])}, muestra[-1])
        model2.learn_one({k: v for k, v in enumerate(muestra[:-1])}, muestra[-1])
        model3.learn_one({k: v for k, v in enumerate(muestra[:-1])}, muestra[-1])
        model4.learn_one({k: v for k, v in enumerate(muestra[:-1])}, muestra[-1])
                            
        if i==999:
            d = {k: v for k, v in enumerate(muestra[:-1])}
            print("Muestra:" + str(d))
    
    
    ent0 = entropia(get_prototypes(model))
    
    
    #MODELO 1, POCAS MUESTRAS DISTINTAS (TOTAL 50 entrenamientos)
    #MODELO 2, MUCHAS MUESTRAS DISTINTAS (TOTAL 5000 entrenamientos)
    #MODELO 3, POCAS MUESTRAS DISTINTAS, MUCHAS ITERACIONES (TOTAL 5000 entrenamientos)
    #MODELO 4, 167 MUESTRAS DISTINTAS, 30 ITERACIONES (TOTAL 167x30-10=5000 entrenamientos)
        
    # Entrenar i veces con j muestras del dataset
    for i in range(1):
        for j in range(50):
            model.learn_one({k: v for k, v in enumerate(dataset.values[j][:-1])}, dataset.values[j][-1])
            
    ent = entropia(get_prototypes(model))
    
    for i in range(1):
        for j in range(5000):
            model2.learn_one({k: v for k, v in enumerate(dataset.values[j][:-1])}, dataset.values[j][-1])
    
    ent2 = entropia(get_prototypes(model2))
    
    for i in range(1000):
        for j in range(5):
            model3.learn_one({k: v for k, v in enumerate(dataset.values[j][:-1])}, dataset.values[j][-1])
    
    ent3 = entropia(get_prototypes(model3))
    
    for i in range(30):
        for j in range(167):
            if i == 29 and j == 157:
                break
            model4.learn_one({k: v for k, v in enumerate(dataset.values[j][:-1])}, dataset.values[j][-1])
            
            
    ent4 = entropia(get_prototypes(model4))
    
    str_ent = f"Entropía inicial: {ent0}\nEntropía modelo 1 (50 muestras, 1 iteración): {ent}\nEntropía modelo 2 (5000 muestras, 1 iteración): {ent2}\
        \nEntropía modelo 3 (5 muestras, 1000 iteraciones): {ent3}\nEntropía modelo 4 (167 muestras, 30 iteraciones, menos 10 entrenamientos): {ent4}\n" 
    print(str_ent)

    
    to_write.append(str_ent)
    
    

    write_to_file(to_write)

if __name__ == "__main__":
    matriz_confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    to_write = []
    
    main()


