import zmq
import numpy as np
import random
import pickle
import multiprocessing
import time
from node_class.deques_proxy import DequeManager
from river import tree

class VFDTreev1:
    
    def __init__(self, id, dataset, nodos=5, s=4, T=1.0, media_llegadas=0.1, puerto_base=10000):
        
        self.id = id
        self.s = min(s, nodos - 1)
        self.T = T        
        self.datalist = [(fila[:-1], fila[-1]) for fila in dataset.values]
        print(f"El nodo {self.id} tiene {len(self.datalist)} muestras.") 
        self.modelo = tree.HoeffdingTreeClassifier()
        self.matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.nodos = nodos
        self.vecinos = [i for i in range(nodos) if i != self.id] if nodos > 1 else []
        self.puerto_base = puerto_base
        self.puertos_vecinos = [self.puerto_base + i for i in self.vecinos]
        self.dir_vecinos = [f"nodo{i}.local" for i in self.vecinos]
        self.manager = DequeManager().start_manager()
        self.lista_modelos = self.manager.DequesProxy(num_deques = self.nodos, maxlen = 1)
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        
        #Atributos auxiliares para obtener estadísticas
        self.muestras_train = 0
        self.params_aggregated = 0
        
        #Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_aggregation = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_espera_total = 0
            
       #Atributos para gestionar hilos/estadisticas
        self.shared_times = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.tiempo_share = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.tiempo_no_share = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.fin_proceso = multiprocessing.Event()
        self.fin_proceso_emisor = multiprocessing.Event()
        self.send_emisor = multiprocessing.Event()
        self.fin_proceso_emisor.clear()
        self.fin_proceso.clear()
        self.send_emisor.clear()
        self.proceso_receptor = multiprocessing.Process(target=self.recibir, args=(self.lista_modelos, self.nodos, self.puerto_base, self.id, self.s, self.T, self.fin_proceso)
                                                        , name=f"Receptor_{self.id}")
        self.proceso_emisor = multiprocessing.Process(target=self.share, args=(self.vecinos, self.send_emisor, self.fin_proceso_emisor, 
                                                                                self.lista_modelos, self.s, self.T, self.shared_times, self.tiempo_share, 
                                                                                self.tiempo_no_share), name=f"Emisor_{self.id}")
        

        
    def run(self):
        
        self.proceso_receptor.start()
        self.proceso_emisor.start()
                    
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.perf_counter()  # Iniciar el temporizador para toda la ejecución.

        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.perf_counter()
            
            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            # while time.perf_counter() - inicio_wait < t_espera:
            #     inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
            #     self.learn_from_queue()  # método para agregar con los parámetros recibidos
            #     self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.

            self.tiempo_espera = time.perf_counter() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.perf_counter()
            self.learn_from_data()
            self.tiempo_learn_data += time.perf_counter() - inicio_learn_data

            self.lista_modelos.appendleft(self.id, self.modelo)
            self.send_emisor.set()

        
        # self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.perf_counter() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        
        self.fin_proceso.set()
        self.fin_proceso_emisor.set()
        self.proceso_receptor.join()
        if self.proceso_receptor.is_alive():
            print(f"El hilo RECEPTOR no ha terminado. Nodo: {self.id}.")

        self.proceso_emisor.join()        
        if self.proceso_emisor.is_alive():
            print(f"El hilo EMISOR ha terminado. Nodo: {self.id}.")
        
        
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
            # f"Ha tardado {self.tiempo_learn_queue / 60} minutos en learn from queue.\n"
            f"Ha tardado {self.tiempo_share_final / 60} minutos en share.\n"
            f"Ha tardado {self.tiempo_no_share_final / 60} minutos en share (No compartiendo).\n")
        
        return
    
    def learn_from_data(self):
        """
        Processes a single data sample from the dataset, updates the model, and evaluates its performance.
        """
        temp = time.perf_counter()
        
        # Increment the sample counter
        self.muestras_train += 1
        
        # Log progress every 250 samples
        print(f"MUESTRA {self.muestras_train}, NODO {self.id}\n") if self.muestras_train % 250 == 0 else None
        
        # Get the next data sample from the dataset
        x, y = self.datalist.pop(0)
        x = {k: v for k, v in enumerate(x)}  # Convert input to a dictionary format if needed
        
        self.lista_modelos.appendleft(self.id, self.modelo)

        prediccion = self.probability_vote(self.get_all_models(), x)
        
        # Update the confusion matrix based on the prediction and true label
        if prediccion == 0 and y == 0:
            self.matriz_conf["TN"] += 1
        elif prediccion == 1 and y == 1:
            self.matriz_conf["TP"] += 1
        elif prediccion == 1 and y == 0:
            self.matriz_conf["FP"] += 1
        elif prediccion == 0 and y == 1:
            self.matriz_conf["FN"] += 1
        else:
            print(f"[NODO {self.id}] Predicción ERRONEA. Predicción: {prediccion}, Etiqueta: {y}")

        # TRAIN: Update the model with the new data sample
        self.modelo.learn_one(x, y)
        
        # Accumulate the time spent processing this sample
        self.tiempo_learn_data += (time.perf_counter() - temp)
    


    def learn_from_queue(self):
        """
        Aggregates the latest parameters received from neighbors and updates the model.
        """
        # Collect the latest parameters from each neighbor's queue
        list_of_params = []
        
        for neighbor_id in self.vecinos:
            if self.cola_params.get_length(neighbor_id) > 0:
                # Retrieve the latest parameters from the neighbor's queue
                params = self.cola_params.getleft(neighbor_id, call_method="LEARNING QUEUE. Retrieving params from neighbor.")
                list_of_params.append(params)
        
        # If no parameters are available from neighbors, skip aggregation
        if not list_of_params:
            print(f"[NODO {self.id}] No hay parámetros nuevos de los vecinos para agregar.")
            return
        
        # Aggregate the parameters and update the model
        print(f"[NODO {self.id}] Agregando parámetros de {len(list_of_params)} vecinos.")
        
        temp = time.perf_counter()

        # Perform weighted aggregation (uniform weights by default)
        self.modelo.aggregate_and_update_parameters(list_of_params)

        self.tiempo_learn_queue += (time.perf_counter() - temp)
        
        print(f"[NODO {self.id}] Modelo actualizado con parámetros agregados.")



    def share(self, vecinos, send_emisor, fin_proceso_emisor, models_list, s, T, shared_times, tiempo_share, tiempo_no_share):
        """
        Method for sharing model parameters with neighbors in a decentralized network.

        Args:
        - vecinos: List of neighbor node IDs.
        - send_emisor: Thread-safe event to signal when to send.
        - fin_proceso_emisor: Thread-safe event to signal the end of the process.
        - models_list: Shared list that contains every tree of the forest.
        - s: Number of neighbors to share with.
        - T: Probability of sharing parameters in each cycle.
        - shared_times: List to track the number of sharing events.
        - tiempo_share: Accumulated time spent in sharing.
        - tiempo_no_share: Accumulated time spent not sharing.
        """
        try:

            # Set up ZeroMQ client socket
            client_context = zmq.Context()
            client_socket = client_context.socket(zmq.ROUTER)
            client_socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB buffer
            client_socket.setsockopt_string(zmq.IDENTITY, f"{self.id}")

            for dir, puerto in zip(self.dir_vecinos, self.puertos_vecinos):
                client_socket.connect(f"tcp://{dir}:{puerto}")

            # Local statistics for sharing
            tiempo_share_local = 0
            shared_times_local = 0
            tiempo_no_share_local = 0

            while True:
                inicio_no_share = time.perf_counter()
                if fin_proceso_emisor.is_set():
                    # Clean up and exit
                    client_socket.setsockopt(zmq.LINGER, 0)
                    client_socket.close()
                    client_context.term()
                    tiempo_share.append(0, tiempo_share_local)
                    print(f"[NODO {self.id}] Se ha añadido: {shared_times_local} a la lista compartida.")
                    shared_times.append(0, shared_times_local)
                    tiempo_no_share.append(0, tiempo_no_share_local)
                    print(f"[NODO {self.id}] El hilo emisor ha terminado.")
                    return

                elif send_emisor.is_set():
                    send_emisor.clear()
                    inicio_share = time.perf_counter()

                    if random.random() < T:  # Probability check for sharing
                        shared_times_local += 1

                        # Get the latest model parameters
                        params_to_share = models_list.getleft(self.id, 0)  # Retrieve the last parameters
                        if params_to_share:
                            # Serialize model parameters for sharing
                            serialized_params = pickle.dumps({
                                "id": self.id,
                                "params": params_to_share
                            })

                            # Randomly select 's' neighbors to share with
                            vecinos_seleccionados = random.sample(vecinos, min(s, len(vecinos)))

                            # Send parameters to the selected neighbors
                            for vecino in vecinos_seleccionados:
                                client_socket.send_multipart([f"{vecino}".encode(), serialized_params])

                    tiempo_share_local += time.perf_counter() - inicio_share  # Update sharing time

                else:
                    time.sleep(0.05)
                    tiempo_no_share_local += time.perf_counter() - inicio_no_share  # Update idle time

        except Exception as e:
            client_socket.setsockopt(zmq.LINGER, 0)
            client_socket.close()
            client_context.term()
            print(f"[ERROR] en SHARE datos en el nodo {self.id}: {e}")
            return





    def recibir(self, models_list, nodos, puerto_base, id, s, T, fin_proceso):
        """
        Function for receiving model parameters asynchronously from neighbors.

        Args:
        - models_list: Shared list that contains every tree of the forest.
        - nodos: Total number of nodes in the network.
        - puerto_base: Base port for receiving messages.
        - id: Node ID.
        - s: Communication frequency.
        - T: Time-related parameter (affects timeout).
        - fin_proceso: Thread-safe flag to signal the end of the process.
        """
        print(f"[NODO {id}] ha iniciado el hilo receptor.")
        if nodos == 1 or s == 0 or T == 0:
            return  # No proceder si solo hay un nodo o no hay comunicación

        # Set up ZMQ server socket
        server_context = zmq.Context()
        server_socket = server_context.socket(zmq.ROUTER)
        server_socket.setsockopt(zmq.RCVBUF, 2000 * 1024 * 1024)  # 2 GB buffer
        server_socket.setsockopt(zmq.IDENTITY, f"{id}".encode())
        server_socket.bind(f"tcp://*:{puerto_base + id}")

        # Set a timeout for receiving messages
        timeout_s = 1
        timeout = int(timeout_s * 1000)  # Convert to milliseconds
        server_socket.setsockopt(zmq.RCVTIMEO, timeout)

        while True:
            try:
                # Block until a message is available
                identidad, mensaje = server_socket.recv_multipart()
                data = pickle.loads(mensaje)

                id_emisor = data["id"]  # Sender node ID
                serialized_params = data["params"]  # Serialized model parameters
                print(f"[NODO {id}] Recibió parámetros del modelo de NODO {id_emisor}.")

                # Add serialized parameters to the deque for processing
                models_list.append(id_emisor, serialized_params)

            except zmq.Again:
                # Timeout handling
                if fin_proceso.is_set():
                    server_socket.setsockopt(zmq.LINGER, 0)
                    server_socket.close()
                    server_context.term()
                    print(f"[NODO {id}] ha terminado de recibir. Vuelve al join.")
                    return

            except Exception as e:
                # General error handling
                server_socket.setsockopt(zmq.LINGER, 0)
                server_socket.close()
                server_context.term()
                print(f"[ERROR] en el nodo {id} al recibir datos: {e}")
                return
            
            
            
    def probability_vote(self, models_list, sample):
        """
        Perform voting based on predicted probabilities, deserializing models inside the function.
        
        Args:
            serialized_models (list): List of serialized models to aggregate.
            sample (dict): The input features for prediction.

        Returns:
            int: The aggregated prediction (0 or 1).
        """
        total_prob = 0
        for i, model in enumerate(models_list):

            prob = model.predict_proba_one(sample).get(1, 0)  # Probability of class 1
            
            # Print the probability predicted by each model
            # print(f"Model {i+1} Probability: {prob:.4f}") if len(models_list) > 1 else None
            total_prob += prob
        
        # Average the probabilities and decide based on a threshold of 0.5
        avg_prob = total_prob / len(models_list)
        return 1 if avg_prob >= 0.5 else 0            
    
    
    
    def get_all_models(self):
        """
        Retrieves all models from the DequesProxy and returns them as a list.
        Each deque contains at most one model.

        Returns:
            list: A list of models (or None if a deque is empty).
        """
        all_models = []
        for i in range(self.nodos):  # Iterate over all deques
            model = self.lista_modelos.getleft(i, call_method="GET_ALL_MODELS")
            if model is not None:
                all_models.append(model)
        return all_models