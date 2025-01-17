import zmq
import numpy as np
import random
import pickle
import multiprocessing
import time
from node_class import DequeManager

class Dsgdv1:
    
    def __init__(self, id, dataset, modelo_grad, modelo_pred=None, share_protocol=None, recomendador=None, nodos=5, s=4, T=0.1, media_llegadas=0.1, puerto_base=10000):
        
        self.id = id
        self.modelo_grad = modelo_grad  # Model for gradient computation
        self.s = min(s, nodos - 1)
        self.T = T        
        self.share_protocol = share_protocol
        self.datalist = [(fila[:-1], fila[-1]) for fila in dataset.values]
        print(f"El nodo {self.id} tiene {len(self.datalist)} muestras.") 
        self.recomendador = recomendador        
        self.matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.nodos = nodos
        self.vecinos = [i for i in range(nodos) if i != self.id] if nodos > 1 else []
        self.puerto_base = puerto_base
        self.puertos_vecinos = [self.puerto_base + i for i in self.vecinos]
        self.dir_vecinos = [f"nodo{i}.local" for i in self.vecinos]
        tam_colas = 500000
        self.manager = DequeManager().start_manager()
        self.cola_grads = self.manager.DequesProxy(num_deques=self.nodos, maxlen=tam_colas)  # Queue for gradients
        self.cola_index = 0
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        
        # Modelo de predicción no siempre es igual al de generación de gradientes
        if modelo_pred:
            self.modelo_pred = modelo_pred
        else:
            self.modelo_pred = modelo_grad
            
        # Atributos auxiliares para obtener estadísticas
        self.muestras_train = 0
        self.grads_train = 0
        self.tam_conj_grads = []
        
        # Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_learn_queue = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_espera_total = 0
            
        # Atributos para gestionar hilos/estadisticas
        self.last_set = self.manager.DequesProxy(num_deques=self.nodos, maxlen=1, id=self.id)
        self.tam_lotes_recibidos = self.manager.ListsProxy(num_lists=1, id=self.id)
        self.compartidos = self.manager.ListsProxy(num_lists=1, id=self.id)
        self.shared_times = self.manager.ListsProxy(num_lists=1, id=self.id)
        self.tiempo_share = self.manager.ListsProxy(num_lists=1, id=self.id)
        self.tiempo_no_share = self.manager.ListsProxy(num_lists=1, id=self.id)
        self.fin_proceso = multiprocessing.Event()
        self.fin_proceso_emisor = multiprocessing.Event()
        self.send_emisor = multiprocessing.Event()
        self.fin_proceso_emisor.clear()
        self.fin_proceso.clear()
        self.send_emisor.clear()
        self.proceso_receptor = multiprocessing.Process(target=self.recibir, args=(self.cola_grads, self.nodos, self.puerto_base, self.id, self.s, self.T, self.fin_proceso, self.tam_lotes_recibidos), name=f"Receptor_{self.id}")
        self.proceso_emisor = multiprocessing.Process(target=self.share, args=(self.id, self.puerto_base, self.vecinos, self.send_emisor, self.fin_proceso_emisor, self.last_set,
                                                                                self.s, self.T, self.shared_times, self.compartidos, self.tiempo_share, 
                                                                                self.tiempo_no_share), name=f"Emisor_{self.id}")
        

    def run(self):
        
        self.proceso_receptor.start()
        self.proceso_emisor.start()
            
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.perf_counter()  # Iniciar el temporizador para toda la ejecución.

        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.perf_counter()
            
            if (self.muestras_train + self.grads_train) % 10 == 0:
                # Añadir tupla a la lista de tamaños de conjunto de gradientes.
                self.tam_conj_grads.append((self.muestras_train + self.grads_train, len(list(self.modelo_grad.buffer.gradients.values()))))

            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            while time.perf_counter() - inicio_wait < t_espera:
                inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
                self.learn_from_queue()  # Método para procesar los gradientes recibidos.
                self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.
                self.save_tam_conj()  # Guardar el tamaño del conjunto de gradientes.

            self.tiempo_espera = time.perf_counter() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.perf_counter()
            self.learn_from_data() 
            self.tiempo_learn_data += time.perf_counter() - inicio_learn_data
            self.save_tam_conj()  # Guardar el tamaño del conjunto de gradientes.
            
            self.last_set.append(self.id, list(self.modelo_grad.buffer.gradients.values()), call_method="RUN. Updating own set for sharing.")
            self.send_emisor.set()


        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.perf_counter() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        # PROBLEMA CON TERMINATE -> NO SE CIERRAN LOS SOCKETS
        join_timeout = 5
        self.fin_proceso.set()
        self.fin_proceso_emisor.set()
        self.proceso_receptor.join()
        if self.proceso_receptor.is_alive():
            print(f"El hilo RECEPTOR no ha terminado. Nodo: {self.id}.")

        self.proceso_emisor.join()        
        if self.proceso_emisor.is_alive():
            print(f"El hilo EMISOR ha terminado. Nodo: {self.id}.")
        
        self.tam_conj_grads = self.diezmar()  # Diezmar la lista de tamaños de conjuntos de gradientes.
        self.tam_lotes_recibidos = self.tam_lotes_recibidos.get_list(0)  # Obtener la lista de tamaños de lotes recibidos.
        
        self.compartidos_final = self.compartidos.pop(0)  # Obtener gradientes compartidos.
        self.shared_times_final = self.shared_times.pop(0)  # Obtener el número de veces que se compartió.
        self.tiempo_share_final = self.tiempo_share.pop(0)  # Obtener el tiempo total de "share".
        self.tiempo_no_share_final = self.tiempo_no_share.pop(0)  # Obtener el tiempo total de "no share".
            
        self.manager.shutdown()
        # Imprimir los tiempos acumulados y el tiempo total de ejecución.
        print(f"[NODO {self.id}. FIN!!].\n"
            f"El tiempo total de espera calculado por muestras ha sido de {sum(self.t_llegadas) / 60} minutos.\n"
            f"Tiempo total de ejecución: {self.tiempo_final_total / 60} minutos.\n"
            f"Tiempo total de espera activa: {self.tiempo_espera_total / 60} minutos.\n"
            f"Ha tardado {self.tiempo_learn_data / 60} minutos en learn from data.\n"
            f"Ha tardado {self.tiempo_learn_queue / 60} minutos en learn from queue.\n"
            f"Ha tardado {self.tiempo_share_final / 60} minutos en share.\n"
            f"Ha tardado {self.tiempo_no_share_final / 60} minutos en share (No compartiendo).\n")
        
        return
        
    def learn_from_data(self):
        
        temp = time.perf_counter()
        
        self.muestras_train += 1
        
        print(f"MUESTRA {self.muestras_train}, NODO {self.id}\n") if self.muestras_train % 250 == 0 else None
        
        x, y = self.datalist.pop(0)            
        x = {k: v for k, v in enumerate(x)}

        # TEST
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
        
        # TRAIN
        self.modelo_grad.learn_one(x, y)
        if not (self.modelo_pred is self.modelo_grad):
            self.modelo_pred.learn_one(x, y)
            
        self.tiempo_learn_data += (time.perf_counter() - temp)
                        
            
    def learn_from_queue(self):
        
        self.update_cola_index()
        colas_revisadas = 0
        colas_a_revisar = self.nodos - 1
        
        while colas_revisadas < colas_a_revisar:
            
            if self.cola_grads.get_length(self.cola_index, call_method="LEARNING QUEUE. Getting number of grads of neighbour.") > 0:
                grads = self.cola_grads.popleft(self.cola_index, call_method="LEARNING QUEUE. Popping grads from neighbour.")
                
                self.grads_train += 1
                print(f"GRAD {self.grads_train}, NODO {self.id}\n") if self.grads_train % 10000 == 0 else None
                
                x = {str(indice): valor for indice, valor in enumerate(grads['x'])}
                y = grads['y']
                
                temp = time.perf_counter()
                # TRAIN
                self.modelo_grad.learn_one(x, y)
                if not (self.modelo_pred is self.modelo_grad):
                    self.modelo_pred.learn_one(x, y)

                self.tiempo_learn_queue += (time.perf_counter() - temp)
                
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
            
        
    def share(self, id, puerto_base, vecinos, send_emisor, fin_proceso_emisor, last_set, s, T, shared_times, compartidos, tiempo_share, tiempo_no_share):
        
        try:
            puertos_vecinos = [puerto_base + i for i in vecinos]
            dir_vecinos = [f"nodo{i}.local" for i in vecinos]

            # Ahora los sockets para enviar
            client_context = zmq.Context()
            client_socket = client_context.socket(zmq.ROUTER)
            client_socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB
            client_socket.setsockopt_string(zmq.IDENTITY, f"{id}")
            
            for dir, puerto in zip(dir_vecinos, puertos_vecinos):
                client_socket.connect(f"tcp://{dir}:{puerto}")


            tiempo_share_local = 0
            shared_times_local = 0
            compartidos_local = 0
            tiempo_no_share_local = 0
            
            while True:
                inicio_no_share = time.perf_counter()
                if fin_proceso_emisor.is_set():
                    client_socket.setsockopt(zmq.LINGER, 0)
                    client_socket.close()
                    client_context.term()
                    tiempo_share.append(0, tiempo_share_local)
                    shared_times.append(0, shared_times_local)
                    print(f"[NODO {id}] . Va a añadir {compartidos_local} a la lista de compartidos.")
                    compartidos.append(0, compartidos_local)
                    print(f"[NODO {id}] . Item añadido: {compartidos.get_item(0, 0)}.")
                    tiempo_no_share.append(0, tiempo_no_share_local)
                    print(f"[NODO {id}] El hilo emisor ha terminado. ORDEN {fin_proceso_emisor} / {fin_proceso_emisor.is_set()}. Vuelve al join.")
                    return
                
                elif send_emisor.is_set():
                    
                    send_emisor.clear()
                    inicio_share = time.perf_counter()

                    if random.random() < T:
                        
                        shared_times_local += 1                         
                        # Obtener los gradientes del modelo
                        grads = last_set.getleft(id, call_method="SHARE. Getting own set for sharing.")

                        # Serializar los datos a compartir
                        grads_to_share = pickle.dumps({"id": id, "grads": [{'x': grad['x'], 'y': grad['y']} for grad in grads]})

                        # Seleccionar aleatoriamente 's' vecinos
                        vecinos_seleccionados = random.sample(vecinos, s)
                        vecinos_eficientes = vecinos_seleccionados
                        sumando = len(grads) * len(vecinos_eficientes)
                        compartidos_local += sumando
                        
                        # Enviar los gradientes a los vecinos seleccionados
                        for vecino in vecinos_eficientes:
                            # Preparar el mensaje para enviar a través de ZeroMQ ROUTER
                            client_socket.send_multipart([f"{vecino}".encode(), grads_to_share])

                    
                    tiempo_share_local += time.perf_counter() - inicio_share  # Acumular tiempo en "share".
        
                else: 
                    time.sleep(0.05)
                    tiempo_no_share_local += time.perf_counter() - inicio_no_share  # Acumular tiempo en "no share".
                
                    
        except Exception as e:
            client_socket.setsockopt(zmq.LINGER, 0)
            client_socket.close()
            client_context.term()
            print(f"[ERROR] en SHARE datos en el nodo {id}: {e}")
            return
        
        
        
    def save_tam_conj(self):
        # Guardar el tamaño cada 10 muestras siempre
        if (self.muestras_train + self.grads_train) % 10 == 0:
            # Calcular el valor actual
            valor_actual = self.muestras_train + self.grads_train
            num_grads = len(list(self.modelo_grad.buffer.gradients.values()))

            # Comprobar si tam_conj_grads no está vacío y el primer elemento de la última tupla es igual a valor_actual
            if not self.tam_conj_grads:
                self.tam_conj_grads.append((valor_actual, num_grads))
            
            elif self.tam_conj_grads[-1][0] != valor_actual:
                # Si la lista está vacía o el valor actual es diferente, añadir la nueva tupla
                self.tam_conj_grads.append((valor_actual, num_grads))


    def diezmar(self):
        datos  = self.tam_conj_grads
        total_muestras = len(datos)
        max_tuplas = 1000  # Número máximo de tuplas deseadas
        
        # Calcular el factor de diezmado necesario. El "+ (total_muestras % max_tuplas > 0)" ajusta para arriba si es necesario
        factor_diezmado = max(total_muestras // max_tuplas, 1) + (total_muestras % max_tuplas > 0)
        
        # Seleccionar datos aplicando el factor de diezmado
        datos_diezmados = [datos[i] for i in range(0, total_muestras, factor_diezmado)]
        
        # Ajustar el resultado para asegurar exactamente 1000 muestras, si el total es mayor que 1000
        if len(datos_diezmados) > max_tuplas:
            datos_diezmados = datos_diezmados[:max_tuplas]
        
        return datos_diezmados