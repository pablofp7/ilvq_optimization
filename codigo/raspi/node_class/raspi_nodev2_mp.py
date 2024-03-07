import zmq
import numpy as np
import random
import pickle
import multiprocessing
import time

class RaspiNodev2_mp:
    
    def __init__(self, manager, id, dataset, modelo_proto, modelo_pred=None, share_protocol=None, recomendador=None, nodos=5, s=4, T=0.1, media_llegadas=0.1, puerto_base=10000):
        
        self.id = id
        self.modelo_proto = modelo_proto
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
        self.manager = manager
        self.cola_protos = manager.DequesProxy(num_deques = self.nodos, maxlen = tam_colas)
        self.cola_index = 0
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        
        #Modelo de predicción no siempre es igual al de generación de prototipos (ILVQ/ILVQ o ARF/ILVQ)
        if modelo_pred:
            self.modelo_pred = modelo_pred
        else:
            self.modelo_pred = modelo_proto
            
        
        #Atributos auxiliares para obtener estadísticas
        self.muestras_train = 0
        self.protos_train = 0
        self.compartidos = 0
        self.shared_times = 0
        self.tam_lotes_recibidos = []
        self.tam_conj_prot = []
        
        #Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_learn_queue = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_share = 0
        self.tiempo_espera_total = 0
            
        #Atributos para gestionar hilos
        self.tam_lotes_recibidos = manager.ListsProxy(num_lists = 1)
        self.fin_proceso = multiprocessing.Event()
        self.proceso_receptor = multiprocessing.Process(target=self.recibir, args=(self.cola_protos, self.nodos, self.puerto_base, self.id, self.s, self.T, self.fin_proceso, self.tam_lotes_recibidos), name=f"Receptor_{self.id}")
        
    def run(self):
        
        self.proceso_receptor.start()
        
        # Ahora los sockets para enviar
        client_context = zmq.Context()
        client_socket = client_context.socket(zmq.ROUTER)
        client_socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB
        client_socket.setsockopt_string(zmq.IDENTITY, f"{self.id}")
        for dir, puerto in zip(self.dir_vecinos, self.puertos_vecinos):
            client_socket.connect(f"tcp://{dir}:{puerto}")
            
            
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.perf_counter()  # Iniciar el temporizador para toda la ejecución.


        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.perf_counter()
            
            if (self.muestras_train + self.protos_train) % 10 == 0:
                #Añadir tupla a la lista de tamaños de conjunto de prototipos. (Muestras train + protos train, tamaño del conjunto de prototipos)
                self.tam_conj_prot.append((self.muestras_train + self.protos_train, len(list(self.modelo_proto.buffer.prototypes.values()))))

            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            while time.perf_counter() - inicio_wait < t_espera:
                inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
                self.learn_from_queue()  # método hipotético para procesar el prototipo
                self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.
                self.save_tam_conj()  # Guardar el tamaño del conjunto de prototipos.

            self.tiempo_espera = time.perf_counter() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.perf_counter()
            self.learn_from_data() 
            self.tiempo_learn_data += time.perf_counter() - inicio_learn_data
            self.save_tam_conj()  # Guardar el tamaño del conjunto de prototipos.

            # Temporizador para "share" después de procesar la muestra.
            inicio_share = time.perf_counter()
            self.share(socket_enviar=client_socket)
            self.tiempo_share += time.perf_counter() - inicio_share  # Acumular tiempo en "share".


        
        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.perf_counter() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        
        self.fin_proceso.set()
        print(f"Fin Proceso set en nodo {self.id}.")
        self.proceso_receptor.join(timeout=10)
        print(f"JOIN en nodo {self.id}.")
        if self.proceso_receptor.is_alive():
            print(f"El proceso receptor en el nodo {self.id} sigue vivo.MATARLO")
            # self.proceso_receptor.terminate()
        
        # Cerramos socket de enviar y contexto
        client_socket.setsockopt(zmq.LINGER, 0)
        client_socket.close()
        client_context.term()
        

        self.tam_conj_prot = self.diezmar()  # Diezmar la lista de tamaños de conjuntos de prototipos.
        self.tam_lotes_recibidos = self.tam_lotes_recibidos.get_list(0)  # Obtener la lista de tamaños de lotes recibidos.


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
        
        temp = time.perf_counter()
        
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
            
        self.tiempo_learn_data += (time.perf_counter() - temp)
                        
            
    def learn_from_queue(self):
        
        self.update_cola_index()
        #Si la cola está vacía, se salta el nodo
        if not self.cola_protos[self.cola_index]:
            return        
        
        # print(f"El nodo {self.id} está procesando la cola {self.cola_index}, que tiene {len(self.cola_protos[self.cola_index])} prototipos.")
        colas_revisadas = 0
        colas_a_revisar = self.nodos - 1
        
        while colas_revisadas < colas_a_revisar:
            
            if self.cola_protos.get_length(self.cola_index) > 0:
                proto = self.cola_protos.popleft(self.cola_index)

                self.protos_train += 1
                print(f"PROTO {self.protos_train}, NODO {self.id}\n") if self.protos_train % 10000 == 0 else None
                
                x = {str(indice): valor for indice, valor in enumerate(proto['x'])}
                y = proto['y']
                
                temp = time.perf_counter()
                #TRAIN
                self.modelo_proto.learn_one(x, y)
                if not (self.modelo_pred is self.modelo_proto):
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
            
        
    def share(self, socket_enviar):
        if random.random() < self.T:
            self.shared_times += 1

            # Obtener los prototipos del modelo
            protos = list(self.modelo_proto.buffer.prototypes.values())
            self.compartidos += len(protos)

            # Serializar los datos a compartir
            proto_to_share = pickle.dumps({"id": self.id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})

            # Seleccionar aleatoriamente 's' vecinos
            vecinos_seleccionados = random.sample(self.vecinos, self.s)

            # Enviar los prototipos a los vecinos seleccionados
            for vecino in vecinos_seleccionados:
                # Preparar el mensaje para enviar a través de ZeroMQ ROUTER
                socket_enviar.send_multipart([bytes(f"{vecino}", encoding="utf-8"), bytes(), proto_to_share])
                                
            return
        
    def save_tam_conj(self):
        # Guardar el tamaño cada 10 muestras siempre
        if (self.muestras_train + self.protos_train) % 10 == 0:
            # Calcular el valor actual
            valor_actual = self.muestras_train + self.protos_train
            num_prototipos = len(list(self.modelo_proto.buffer.prototypes.values()))

            # Comprobar si tam_conj_prot no está vacío y el primer elemento de la última tupla es igual a valor_actual
            if not self.tam_conj_prot:
                self.tam_conj_prot.append((valor_actual, num_prototipos))
            
            elif self.tam_conj_prot[-1][0] != valor_actual:
                # Si la lista está vacía o el valor actual es diferente, añadir la nueva tupla
                self.tam_conj_prot.append((valor_actual, num_prototipos))


    def diezmar(self):
        datos  = self.tam_conj_prot
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


    def recibir(uk_arg, cola_protos, nodos, puerto_base, id, s, T, fin_proceso, tam_lotes_recibidos):
        
        # #Vamos a printear todos lso argumentos:
        # print(f"")
        # print(f"Argumentos del método recibir:")
        # print(f"unknown arg: {uk_arg}")
        # print(f"cola_protos: {cola_protos}")
        # print(f"nodos: {nodos}")
        # print(f"puerto_base: {puerto_base}")
        # print(f"id: {id}")
        # print(f"s: {s}")
        # print(f"T: {T}")
        # print(f"fin_proceso: {fin_proceso}")
        # print(f"tam_lotes_recibidos: {tam_lotes_recibidos}")
        # print(f"")
        
        if nodos == 1 or s == 0 or T == 0:
            return  # No proceder si so lo hay un nodo o no hay comparticion


        server_context = zmq.Context()
        server_socket = server_context.socket(zmq.ROUTER)
        server_socket.setsockopt(zmq.RCVBUF, 2000 * 1024 * 1024)  # 2 GB
        server_socket.setsockopt(zmq.IDENTITY, f"{id}".encode())
        server_socket.bind(f"tcp://*:{puerto_base + id}")
        # El timeout depende de T, porque con T bajo, el nodo debe esperar más tiempo
        timeout_s = 1
        timeout = int(timeout_s * 1000)  # Convertir a milisegundos
        server_socket.setsockopt(zmq.RCVTIMEO, timeout)  # Establecer un tiempo de espera para el socket
        
        while True:
            try:
                # print(f"[RECIBIR-{id}] Esperando por mensaje...")
                # Bloquear hasta que un mensaje esté disponible
                identidad, mensaje = server_socket.recv_multipart()
                # id_emisor = identidad.decode()  # Convertir la identidad del emisor de bytes a str si es necesario
                
                data = pickle.loads(mensaje)
                id_recibido = data["id"]  
                protos = data["protos"]
                
                # print(f"El nodo {id} ha recibido {len(protos)} prototipos del nodo {id_recibido}.")
                            
                # Procesar los prototipos recibidos
                # Por ejemplo, añadir los prototipos recibidos a la cola correspondiente para su procesamiento
                # print(f"[RECIBIR-{id}] Añadiendo {len(protos)} prototipos a la cola del nodo {id}.")
                cola_protos.extendleft(id_recibido, protos)
                # print(f"[RECIBIR-{id}] Prototipos añadidos a la cola del nodo {id}.")
                tam_lotes_recibidos.append(0, (id_recibido, len(protos)))
                # print(f"[RECIBIR-{id}] Tamaño de lotes recibidos actualizado.")

    
            # except zmq.ContextTerminated:
            #     print(f"El contexto de ZeroMQ ha terminado en el nodo {id}.")
            #     server_socket.setsockopt(zmq.LINGER, 0)
            #     server_socket.close()   
            #     server_context.term()
            #     return
            
            except zmq.Again:
                # print(f"El nodo {id} ha agotado el tiempo de espera.")
                if fin_proceso.is_set():
                    server_socket.setsockopt(zmq.LINGER, 0)
                    server_socket.close()
                    server_context.term()
                    print(f"El nodo {id} ha terminado de recibir.")
                    return
                # else:
                #     print(f"El nodo {id} lleva {timeout_s} segundos esperando. Pero no ha acabado de recibir")
            
            except Exception as e:
                server_socket.setsockopt(zmq.LINGER, 0)
                server_socket.close()   
                server_context.term()
                print(f"Error al recibir datos en el nodo {id}: {e}")
                return





        
