import zmq
import numpy as np
import random
import pickle
import multiprocessing
import time
from collections import deque
from entropia import jsd

class RaspiNodev4_1local_mp:
    
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
        tam_colas = 500000
        self.manager = manager
        self.proxy_deques = manager.DequesProxy(num_deques = self.nodos, maxlen = tam_colas)        
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
        self.tam_lotes_recibidos = []
        self.tam_conj_prot = []

        
        #Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_learn_queue = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_espera_total = 0
        
        #Atributos para gestionar hilos
        self.conjuntos_prototipos = manager.DequesProxy(num_deques = self.nodos, maxlen = 1)
        self.tam_lotes_recibidos = manager.ListsProxy(num_lists = 1)
        self.compartidos = manager.ListsProxy(num_lists = 1)
        self.shared_times = manager.ListsProxy(num_lists = 1)
        self.tiempo_share = manager.ListsProxy(num_lists = 1)
        self.fin_proceso = multiprocessing.Event()
        self.proceso_receptor = multiprocessing.Process(target=self.recibir, args=(self.proxy_deques, self.nodos, self.puerto_base, self.id, self.s, self.T, self.fin_proceso, self.tam_lotes_recibidos, self.conjuntos_prototipos), name=f"Receptor_{self.id}")
        self.fin_proceso_emisor = multiprocessing.Event()
        self.send_emisor = multiprocessing.Event()
        self.proceso_emisor = multiprocessing.Process(target=self.share, name = f"Emisor_{self.id}", args=(self.id, self.puerto_base, self.vecinos, self.send_emisor,
                                                    self.fin_proceso_emisor, self.conjuntos_prototipos, self.s, self.T, self.shared_times, self.compartidos, self.tiempo_share))
            
        
    def run(self):
        
        self.proceso_receptor.start()
        self.proceso_emisor.start()            
            
        print(f"Inicia ejecución del nodo {self.id}")
        tiempo_inicio_total = time.perf_counter()  # Iniciar el temporizador para toda la ejecución.


        # Iterar sobre cada muestra y su tiempo de espera correspondiente.
        for t_espera in self.t_llegadas:
            inicio_wait = time.perf_counter()
            
            # Esperamos el tiempo designado, pero mientras esperamos, continuamos procesando la cola.
            while time.perf_counter() - inicio_wait < t_espera:
                inicio_procesamiento = time.perf_counter_ns()  # Iniciar el temporizador para el procesamiento.
                self.learn_from_queue()  # método hipotético para procesar el prototipo
                self.tiempo_learn_queue += time.perf_counter_ns() - inicio_procesamiento  # Acumular tiempo.
                self.save_tam_conj()

            self.tiempo_espera = time.perf_counter() - inicio_wait
            self.tiempo_espera_total += self.tiempo_espera  # Acumular el tiempo real de espera.

            # Después de esperar, procesamos la muestra actual del dataset.
            inicio_learn_data = time.perf_counter()
            self.learn_from_data() 
            self.tiempo_learn_data += time.perf_counter() - inicio_learn_data
            self.save_tam_conj()

            self.conjuntos_prototipos.append(self.id, list(self.modelo_proto.buffer.prototypes.values()))
            self.send_emisor.set()
            
        
        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.perf_counter() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        
        # PROBLEMA CON TERMINATE -> NO SE CIERRAN LOS SOCKETS
        self.fin_proceso.set()
        self.fin_proceso_emisor.set()
        self.proceso_receptor.join(timeout=5)
        if self.proceso_receptor.is_alive():
            self.proceso_receptor.terminate()
            print(f"El hilo RECEPTOR no ha terminado. Nodo: {self.id}.")

        self.proceso_emisor.join(timeout=5)        
        if self.proceso_emisor.is_alive():
            print(f"El hilo EMISOR ha terminado. Nodo: {self.id}.")
            self.proceso_emisor.terminate()
        
        self.tam_conj_prot = self.diezmar()  # Diezmar la lista de tamaños de conjuntos de prototipos.
        self.tam_lotes_recibidos = self.tam_lotes_recibidos.get_list(0)  # Obtener la lista de tamaños de lotes recibidos.
        try:
            self.compartidos = self.compartidos.pop(0) # Obtener prototipos compartidos.
        except:
            print(f"0 COMPARTIDOS")
            self.compartidos = 0
            
        try: 
            self.shared_times = self.shared_times.pop(0)  # Obtener el número de veces que se compartió.
        except:
            print(f"0 SHARED TIMES")
            self.shared_times = 0
            
        try:
            self.tiempo_share = self.tiempo_share.pop(0)  # Obtener el tiempo total de "share".
        except:
            print(f"0 TIEMPO SHARE")
            self.tiempo_share = 0
            
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
        
        # print(f"El nodo {self.id} está procesando la cola {self.cola_index}, que tiene {len(self.cola_protos[self.cola_index])} prototipos.")
        colas_revisadas = 0
        colas_a_revisar = self.nodos - 1
        
        while colas_revisadas < colas_a_revisar:
            
            if self.proxy_deques.get_length(self.cola_index) > 0:
                proto = self.proxy_deques.popleft(self.cola_index)
                
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
            
        
    def share(self, id, puerto_base, vecinos, send_emisor, fin_proceso_emisor, colas_conj, s, T, shared_times, compartidos, tiempo_share):
        
        puertos_vecinos = [puerto_base + i for i in vecinos]
        # Ahora los sockets para enviar
        client_context = zmq.Context()
        client_socket = client_context.socket(zmq.ROUTER)
        client_socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB
        client_socket.setsockopt_string(zmq.IDENTITY, f"{id}")

        for puerto in puertos_vecinos:
            client_socket.connect(f"tcp://localhost:{puerto}")

        while True:
            
            if fin_proceso_emisor.is_set():
                print(f"El hilo emisor ha terminado. Nodo: {id}.")
                client_socket.setsockopt(zmq.LINGER, 0)
                client_socket.close()
                client_context.term()
                return
            
            if send_emisor.is_set():
                
                send_emisor.clear()
                inicio_share = time.perf_counter()

                if random.random() < T:
                    
                    if shared_times.get_length(0) > 0:
                        shar_prev = shared_times.pop(0)
                    else:
                        shar_prev = 0
                        
                    print(f"El nodo {id} ha compartido {shar_prev + 1} veces.") if id == 0 else None
                    shared_times.append(0, shar_prev + 1)
                    # Obtener los prototipos del modelo
                    protos = colas_conj.getleft(id)

                    # Serializar los datos a compartir
                    proto_to_share = pickle.dumps({"id": id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})

                    # Seleccionar aleatoriamente 's' vecinos
                    vecinos_seleccionados = random.sample(vecinos, s)
                    
                    vecinos_eficientes = self.check_sharing(vecinos_seleccionados, colas_conj, id)
                    
                    sumando = len(protos) if vecinos_eficientes else 0
                    if compartidos.get_length(0) > 0:
                        comp_previo = compartidos.pop(0)
                    else: 
                        comp_previo = 0
                        
                    print(f"El nodo {id} ha compartido {comp_previo + sumando} prototipos.") if id == 0 else None
                    compartidos.append(0, comp_previo + sumando)
                    
                    # Enviar los prototipos a los vecinos seleccionados
                    for vecino in vecinos_eficientes:
                        # Preparar el mensaje para enviar a través de ZeroMQ ROUTER
                        client_socket.send_multipart([f"{vecino}".encode(), proto_to_share])

                
                t_share = time.perf_counter() - inicio_share  # Acumular tiempo en "share".
                if tiempo_share.get_length(0) > 0:
                    previo = tiempo_share.pop(0)
                else:
                    previo = 0
                
                print(f"Tiempo Acumulado en Share: {previo + t_share}.") if id == 0 else None
                tiempo_share.append(0, previo + t_share)

                
    
    def check_sharing(self, destinos, colas_conj, id):
        
        mi_conj = colas_conj.getleft(id)
        mi_conj = np.array([np.append(proto['x'], proto['y']) for proto in mi_conj])
        # self.conj_prot[self.id].append(mis_protos)

        
        destinos_eficiente = []
        for destino in destinos:
            if colas_conj.get_length(destino) < 1:
                destinos_eficiente.append(destino)
                print(f"El nodo {self.id} comparte con el nodo {destino} porque no tiene prototipos.") 
                continue
            
            #Vamos a printear los conjuntos de prototipos a evaluar
            dest_conj = colas_conj.getleft(destino)
            # print(f"Mi conjunto: {mi_conj}.")
            # print(f"Conjunto del nodo {destino}: {dest_conj}.")
            distancia = jsd.monte_carlo_jsd(mi_conj, dest_conj)
            print(f"Distancia entre conjuntos de prototipos del nodo {self.id} y el nodo {destino}: {distancia}.")  if self.id == 0 else None
            if distancia > 0.2:
                print(f"Es eficiente compartirle al nodo {destino}.") if self.id == 0 else None
                destinos_eficiente.append(destino)
            
            print(f"Los vecinos eficientes son: {destinos_eficiente}.") if self.id == 0 else None
                
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

   

    def recibir(self, proxy_deques, nodos, puerto_base, id, s, T, fin_proceso, tam_lotes_recibidos, colas_conj):
        if nodos == 1 or s == 0 or T == 0:
            return  # No proceder si solo hay un nodo o no hay comparticion

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
                # Bloquear hasta que un mensaje esté disponible
                identidad, mensaje = server_socket.recv_multipart()
                # id_emisor = identidad.decode()  # Convertir la identidad del emisor de bytes a str si es necesario
                
                data = pickle.loads(mensaje)
                id_recibido = data["id"]  
                protos = data["protos"]
                
                # print(f"El nodo {id} ha recibido {len(protos)} prototipos del nodo {id_recibido}.")
                            
                # Procesar los prototipos recibidos
                # Por ejemplo, añadir los prototipos recibidos a la cola correspondiente para su procesamiento
                proxy_deques.extendleft(id_recibido, protos)
                self.update_conj_proto(id_recibido, protos, colas_conj)
                tam_lotes_recibidos.append(0, (id_recibido, len(protos)))

    
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
                print(f"Error al recibir datos en el nodo {self.id}: {e}")
                return
        

        
    def update_conj_proto(self, id, protos, colas_conj):
        protos_transformed = np.array([np.append(proto['x'], proto['y']) for proto in protos])
        colas_conj.append(id, protos_transformed)
        
        
        
