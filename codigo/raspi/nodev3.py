import sysv_ipc
import numpy as np
import random
import pickle
import threading
import time
from collections import deque

class Nodev3:
    
    def __init__(self, id, dataset, modelo_proto, modelo_pred = None, share_protocol=None, recomendador=None, nodos=5, s = 4, T = 0.1, media_llegadas = 0.1):
        
        self.id = id
        self.modelo_proto = modelo_proto
        self.s = min(s, nodos - 1)
        self.T = T        
        self.share_protocol = share_protocol
        #Se pasan la filas del dataframe a listas x,y donde 'x' es una lista con la variables de entrada e 'y' es la etiqueta correspondiente
        self.datalist = [(fila[:-1], fila[-1]) for fila in dataset.values] 
        print(f"El nodo {self.id} tiene {len(self.datalist)} muestras.") 
        self.recomendador = recomendador        
        self.matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.nodos = nodos
        self.vecinos = [i for i in range(nodos) if i != self.id] if nodos > 1 else []
        self.colas = [sysv_ipc.MessageQueue(key=id + 10, flags=sysv_ipc.IPC_CREAT, max_message_size=100000) for id in range(nodos)]
        self.cola_protos = [deque(maxlen=100000) for _ in range(self.nodos)]
        self.cola_index = 0
        
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        

        #Modelo de predicción no siempre es igual al de generación de prototipos (ILVQ/ILVQ o ARF/ILVQ)
        if modelo_pred:
            self.modelo_pred = modelo_pred
        else:
            self.modelo_pred = modelo_proto
            
        #Atributos para controlar el fin de la ejecución
        self.fin_hilo = False

        #Atributos para gestionar hilos
        self.hilo_receptor = threading.Thread(target=self.recibir)
        self.lock_cola = threading.Lock()
        
        #Atributos auxiliares para obtener estadísticas
        self.muestras_train = 0
        self.protos_train = 0
        self.compartidos = 0
        self.shared_times = 0
        self.tam_lotes_recibidos = []
        
        #Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_learn_queue = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_share = 0
        self.tiempo_espera_total = 0
        
    
        
    def run(self):
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.time()  # Iniciar el temporizador para toda la ejecución.

        self.hilo_receptor.start()

        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.time()

            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            while time.time() - inicio_wait < t_espera:
                inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
                if self.cola_protos:  # si hay prototipos en la cola
                    self.learn_from_queue()  # método hipotético para procesar el prototipo
                self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.

            self.tiempo_espera = time.time() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.time()
            self.learn_from_data() 
            self.tiempo_learn_data += time.time() - inicio_learn_data

            # Temporizador para "share" después de procesar la muestra.
            inicio_share = time.time()
            self.share()
            self.tiempo_share += time.time() - inicio_share  # Acumular tiempo en "share".

        self.fin_hilo = True

        print(f"Esperando a que acabe el hilo de recibir en el nodo {self.id}")
        self.hilo_receptor.join()
        
        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.time() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.

        # Imprimir los tiempos acumulados y el tiempo total de ejecución.
        print(f" - El nodo {self.id} ha terminado de ejecutar TODO.\n"
            f"El tiempo total de espera calculado por muestras ha sido de {sum(self.t_llegadas) / 60} minutos.\n"
            f"Tiempo total de ejecución: {self.tiempo_final_total / 60} minutos.\n"
            f"Tiempo total de espera activa: {self.tiempo_espera_total / 60} minutos.\n"
            f"Ha tardado {self.tiempo_learn_data / 60} minutos en learn from data.\n"
            f"Ha tardado {self.tiempo_learn_queue / 60} minutos en learn from queue.\n"
            f"Ha tardado {self.tiempo_share / 60} minutos en share.\n")  # <-- Añadir tiempo de "share".
        
        return

        
    def learn_from_data(self):
        
        temp = time.time()
        
        self.muestras_train += 1
        
        print(f"MUESTRA {self.muestras_train}, NODO {self.id}\n") if self.muestras_train % 250 == 0 else None
        
        x, y = self.datalist.pop(0)            
        x = {k: v for k, v in enumerate(x)}

        #TEST
        prediccion = self.modelo_pred.predict_one(x)
        if isinstance(prediccion, dict):
            if 1.0 in prediccion:
                prediccion = prediccion[1.0]
            else:
                prediccion = 0.0      

        if prediccion == 0 and y == 0:
            self.matriz_conf["TN"] += 1
        elif prediccion == 1 and y == 1:
            self.matriz_conf["TP"] += 1
        elif prediccion == 1 and y == 0:
            self.matriz_conf["FP"] += 1
        else:
            self.matriz_conf["FN"] += 1
        
        #TRAIN
        self.modelo_proto.learn_one(x, y)
        if not (self.modelo_pred is self.modelo_proto):
            self.modelo_pred.learn_one(x, y)
            
        self.tiempo_learn_data += (time.time() - temp)
                        
            
    def learn_from_queue(self):
        
        self.update_cola_index()
        colas_revisadas = 0
        colas_a_revisar = self.nodos - 1
        
        while colas_revisadas < colas_a_revisar:
            
            cola_actual = self.cola_protos[self.cola_index]
            
            if cola_actual:
                proto = cola_actual.popleft()
                self.protos_train += 1
                print(f"PROTO {self.protos_train}, NODO {self.id}\n") if self.protos_train % 10000 == 0 else None
                
                x = {str(indice): valor for indice, valor in enumerate(proto['x'])}
                y = proto['y']
                
                temp = time.time()
                #TRAIN
                self.modelo_proto.learn_one(x, y)
                if not (self.modelo_pred is self.modelo_proto):
                    self.modelo_pred.learn_one(x, y)

                self.tiempo_learn_queue += (time.time() - temp)
                
                return
            
            self.update_cola_index()
            colas_revisadas += 1


    def update_cola_index(self):
        """
        Actualiza el índice de la cola, saltándose la cola del nodo actual.
        """
        self.cola_index = (self.cola_index + 1) % self.nodos
        if self.cola_index == self.id:
            self.cola_index = (self.cola_index + 1) % self.nodos
    
    def share(self, tipo='proto'):
        
        if "proto" in tipo:
            #Sólo se comparte T% de las veces
            if random.random() < self.T:
                
                self.shared_times += 1
                
                #Se obtienen los prototipos del modelo
                protos = list(self.modelo_proto.buffer.prototypes.values())
                self.compartidos += len(protos)
                
                # print(f"El nodo 1 va a compartir {len(protos)} prototipos.") if self.id == 0 else None
                
                proto_to_share = pickle.dumps({"id": self.id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})
            
                #Se seleccionan aleatoriamente 's' vecinos
                vecinos_seleccionados = random.sample(self.vecinos, self.s)
                
                # for proto in protos:
                #     for vecino in vecinos_seleccionados:
                #         #Se formatea cada prototipo para enviar solo los campos necesarios ('x' e 'y')
                #         proto_aux = {'x': proto['x'], 'y': proto['y']}
                #         self.colas[vecino].send(pickle.dumps(proto_aux), type=1, block = False)
                
                for vecino in vecinos_seleccionados:   
                    try:
                        self.colas[vecino].send(proto_to_share, type=1, block = False)
                    except Exception as e:
                        print(f"Demasiado grande el mensaje: {len(proto_to_share)}")        
                # print(f"El nodo {self.id} ha enviado {len(protos)} prototipos.")
                
        elif "fin" in tipo:
            # print(f"El nodo {self.id} ha terminado el dataset, deja de compartir. Enviar mensaje tipo 2 al resto de nodos.")
            for vecino in self.vecinos:
                self.colas[vecino].send(str(self.id), type=2, block = False)
            
        return
    

    def recibir(self):
    
        if self.nodos == 1:
            return
        
        while True:
            
            if self.fin_hilo:
                return
            
            try:
                msg, t = self.colas[self.id].receive(block = False)
                if t == 1:
                    
                    temp = time.time()
                    msg = pickle.loads(msg)
                    id_recibido = msg["id"]
                    protos = msg["protos"]
                    self.tam_lotes_recibidos.append((id_recibido, len(protos)))
                     
                    self.cola_protos[id_recibido].extendleft(protos)
                    self.t_queue_recibir += (time.time() - temp)
                    # print(f"El nodo {self.id} ha recibido un prototipo del nodo {id_recibido}.") if self.id == 0 else None
                        
            except sysv_ipc.BusyError:
                if self.colas[self.id].last_receive_time != 0:
                    t_sin_recibir = time.time()  - self.colas[self.id].last_receive_time
                    # print(f"Tiempo sin recibir en el nodo {self.id}: {t_sin_recibir}, tiempo actual: {time.time()}, ultima recepción: {self.colas[self.id].last_receive_time}")
                    if (t_sin_recibir % 10) == 0:
                        print(f"El nodo {self.id} lleva {t_sin_recibir} segundos sin recibir nada.")
                        # return
                
            # time.sleep(0.01)
            
        # print(f"Saliendo de hilo de recepcion NODO -> {self.id}")
        