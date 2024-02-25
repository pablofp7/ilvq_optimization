import zmq
import numpy as np
import random
import pickle
import threading
import time
from collections import deque
from entropia import jsd

class RaspiNodev4:
    
    def __init__(self, id, dataset, modelo_proto, modelo_pred=None, share_protocol=None, recomendador=None, nodos=5, s=4, T=0.1, media_llegadas=0.1, puerto_base=10000):
        
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
        self.cola_protos = [deque(maxlen=100000) for _ in range(self.nodos)]
        self.cola_index = 0
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        
        self.conj_prot = [deque(maxlen=1) for _ in range(self.nodos)]
        
        #Modelo de predicción no siempre es igual al de generación de prototipos (ILVQ/ILVQ o ARF/ILVQ)
        if modelo_pred:
            self.modelo_pred = modelo_pred
        else:
            self.modelo_pred = modelo_proto
            

        #Atributos para gestionar hilos
        self.hilo_receptor = threading.Thread(target=self.recibir)
        self.fin_hilo = False
        
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
            
        
    def run(self):
        
        self.hilo_receptor.start()
        # Ahora los sockets para enviar
        client_context = zmq.Context()
        client_socket = client_context.socket(zmq.ROUTER)
        client_socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB
        client_socket.setsockopt_string(zmq.IDENTITY, f"{self.id}")
        for dir, puerto in zip(self.dir_vecinos, self.puertos_vecinos):
            client_socket.connect(f"tcp://{dir}:{puerto}")
            
            
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.time()  # Iniciar el temporizador para toda la ejecución.


        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.time()
            
            if (self.muestras_train + self.protos_train) % 10 == 0:
                #Añadir tupla a la lista de tamaños de conjunto de prototipos. (Muestras train + protos train, tamaño del conjunto de prototipos)
                self.tam_conj_prot.append((self.muestras_train + self.protos_train, len(list(self.modelo_proto.buffer.prototypes.values()))))

            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            while time.time() - inicio_wait < t_espera:
                inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
                self.learn_from_queue()  # método hipotético para procesar el prototipo
                self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.
                self.save_tam_conj()  # Guardar el tamaño del conjunto de prototipos.

            self.tiempo_espera = time.time() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.time()
            self.learn_from_data() 
            self.tiempo_learn_data += time.time() - inicio_learn_data
            self.save_tam_conj()  # Guardar el tamaño del conjunto de prototipos.

            # Temporizador para "share" después de procesar la muestra.
            inicio_share = time.time()
            self.share(socket_enviar=client_socket)
            self.tiempo_share += time.time() - inicio_share  # Acumular tiempo en "share".


        
        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.time() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        
        self.fin_hilo = True
        self.hilo_receptor.join()
        
        # Cerramos socket de enviar y contexto
        client_socket.setsockopt(zmq.LINGER, 0)
        client_socket.close()
        client_context.term()
        

        self.tam_conj_prot = self.diezmar()  # Diezmar la lista de tamaños de conjuntos de prototipos.

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
        #Si la cola está vacía, se salta el nodo
        if not self.cola_protos[self.cola_index]:
            return        
        
        # print(f"El nodo {self.id} está procesando la cola {self.cola_index}, que tiene {len(self.cola_protos[self.cola_index])} prototipos.")
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
            
            vecinos_eficientes = self.check_sharing(vecinos_seleccionados)

            # Enviar los prototipos a los vecinos seleccionados
            for vecino in vecinos_eficientes:
                # Preparar el mensaje para enviar a través de ZeroMQ ROUTER
                socket_enviar.send_multipart([bytes(f"{vecino}", encoding="utf-8"), bytes(), proto_to_share])
                                
            return
        
        
    def check_sharing(self, destinos):
        
        mis_protos = np.array([np.append(proto['x'], proto['y']) for proto in list(self.modelo_proto.buffer.prototypes.values())])
        print(f"Mis protos: {mis_protos[:10]}") if self.id == 0 else None
        self.conj_prot[self.id].append(mis_protos)

        destinos_eficiente = []
        for destino in destinos:
            print(f"La cola de prototipos del nodo {destino} tiene: {self.conj_prot[destino]}.")  if self.id == 0 else None
            print(f"Mi cola de prototipos tiene: {self.conj_prot[self.id]}.")  if self.id == 0 else None
            if not self.conj_prot[destino]:
                destinos_eficiente.append(destino)
                print(f"El nodo {self.id} comparte con el nodo {destino} porque no tiene prototipos.")  if self.id == 0 else None
                continue
            print(f"Formato de conjuntos de prototipos:\nMIO: {self.conj_prot[self.id][0][:10]}.\nVECINO: {self.conj_prot[destino][0][:10]}.")  if self.id == 0 else None
            distancia = jsd.monte_carlo_jsd(self.conj_prot[self.id][0], self.conj_prot[destino][0])
            print(f"Distancia entre conjuntos de prototipos del nodo {self.id} y el nodo {destino}: {distancia}.")  if self.id == 0 else None
            if distancia > 0.5:
                destinos_eficiente.append(destino)
                
        return destinos_eficiente



        
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


    def recibir(self):
        if self.nodos == 1 or self.s == 0 or self.T == 0:
            return  # No proceder si solo hay un nodo o no hay comparticion

        server_context = zmq.Context()
        server_socket = server_context.socket(zmq.ROUTER)
        server_socket.setsockopt(zmq.RCVBUF, 2000 * 1024 * 1024)  # 2000 MB
        server_socket.setsockopt_string(zmq.IDENTITY, f"{self.id}")
        server_socket.bind(f"tcp://*:{self.puerto_base + self.id}")
        # El timeout depende de T, porque con T bajo, el nodo debe esperar más tiempo
        
        timeout_s = 1
        timeout = int(timeout_s * 1000)  # Convertir a milisegundos
        server_socket.setsockopt(zmq.RCVTIMEO, timeout)  # Establecer un tiempo de espera para el socket
        
        while not self.fin_hilo:
            try:
                # Bloquear hasta que un mensaje esté disponible
                identidad, _, mensaje = server_socket.recv_multipart()
                # id_emisor = identidad.decode()  # Convertir la identidad del emisor de bytes a str si es necesario
                
                data = pickle.loads(mensaje)
                id_recibido = data["id"]  
                protos = data["protos"]
                
                # print(f"El nodo {self.id} ha recibido {len(protos)} prototipos del nodo {id_recibido}.")
                            
                # Procesar los prototipos recibidos
                # Por ejemplo, añadir los prototipos recibidos a la cola correspondiente para su procesamiento
                self.cola_protos[id_recibido].extendleft(protos)
                self.update_conj_proto(id_recibido, protos)
                self.tam_lotes_recibidos.append((id_recibido, len(protos)))

    
            except zmq.ContextTerminated:
                print(f"El contexto de ZeroMQ ha terminado en el nodo {self.id}.")
                server_socket.setsockopt(zmq.LINGER, 0)
                server_socket.close()   
                server_context.term()

                return
            except zmq.Again:
                # print(f"El nodo {self.id} ha agotado el tiempo de espera.")
                if self.fin_hilo:
                    server_socket.setsockopt(zmq.LINGER, 0)
                    server_socket.close()
                    server_context.term()
                    print(f"El nodo {self.id} ha terminado de recibir.")
                    return
                # else:
                #     print(f"El nodo {self.id} lleva {timeout_s} segundos esperando. Pero no ha acabado de recibir")
            
            except Exception as e:
                server_socket.setsockopt(zmq.LINGER, 0)
                server_socket.close()   
                server_context.term()
                print(f"Error al recibir datos en el nodo {self.id}: {e}")
                return
        
        server_socket.setsockopt(zmq.LINGER, 0)
        server_socket.close()
        server_context.term()
       
        
    def update_conj_proto(self, id, protos):
        protos_transformed = np.array([np.append(proto['x'], proto['y']) for proto in protos])
        self.conj_prot[id].append(protos_transformed)
        print(f"El nodo {self.id} ha actualizado el conjunto de prototipos del nodo {id}.") if self.id == 0 else None


        
