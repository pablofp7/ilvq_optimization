import zmq
import numpy as np
import random
import pickle
import multiprocessing
import time
from collections import deque
from entropia import jsd
from node_class.deques_proxy import DequeManager

class Nodev4:
    
    def __init__(self, id, dataset, modelo_proto, modelo_pred=None, share_protocol=None, recomendador=None, nodos=5, s=4, T=0.1, media_llegadas=0.1, puerto_base=10000, tam_colas = 1):
        
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
        self.tam_colas = tam_colas
        self.manager = DequeManager().start_manager()
        self.cola_protos = self.manager.DequesProxy(num_deques = self.nodos, maxlen = tam_colas)
        self.cola_index = 0
        self.t_llegadas = np.random.exponential(media_llegadas, len(self.datalist)).tolist()
        
        self.conj_prot = [deque(maxlen=1) for _ in range(self.nodos)]
        
        #Modelo de predicción no siempre es igual al de generación de prototipos (ILVQ/ILVQ o ARF/ILVQ)
        if modelo_pred:
            self.modelo_pred = modelo_pred
        else:
            self.modelo_pred = modelo_proto
            

        
        #Atributos auxiliares para obtener estadísticas
        self.muestras_train = 0
        self.protos_train = 0
        self.tam_conj_prot = []
        self.clust_time = 0
        self.clust_runs = 0
        
        #Medidores de tiempo
        self.tiempo_learn_data = 0
        self.tiempo_learn_queue = 0
        self.t_queue_lock = 0
        self.t_queue_recibir = 0
        self.t_queue_pickle = 0
        self.tiempo_final_total = 0
        self.tiempo_espera_total = 0
            
       #Atributos para gestionar hilos/estadisticas
        self.last_set = self.manager.DequesProxy(num_deques = self.nodos, maxlen = 1, id = self.id)
        self.tam_lotes_recibidos = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.compartidos = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.shared_times = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.tiempo_share = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.tiempo_no_share = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.no_comp_jsd = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.protos_descartados = self.manager.ListsProxy(num_lists = 1, id = self.id)
        self.fin_proceso = multiprocessing.Event()
        self.fin_proceso_emisor = multiprocessing.Event()
        self.send_emisor = multiprocessing.Event()
        self.fin_proceso_emisor.clear()
        self.fin_proceso.clear()
        self.send_emisor.clear()
        self.proceso_receptor = multiprocessing.Process(target=self.recibir, args=(self.cola_protos, self.nodos, self.puerto_base, self.id, self.s, self.T, self.fin_proceso, 
                                                                                   self.tam_lotes_recibidos,self.last_set, self.protos_descartados), name=f"Receptor_{self.id}")
        self.proceso_emisor = multiprocessing.Process(target=self.share, args=(self.id, self.puerto_base, self.vecinos, self.send_emisor, self.fin_proceso_emisor, 
                                                                                self.last_set, self.s, self.T, self.shared_times, self.compartidos, self.tiempo_share, 
                                                                                self.tiempo_no_share, self.no_comp_jsd), name=f"Emisor_{self.id}")
        

        
    def run(self):
        
        self.proceso_receptor.start()
        self.proceso_emisor.start()
                    
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

            self.last_set.append(self.id, list(self.modelo_proto.buffer.prototypes.values()), call_method = "RUN. Updating own set for sharing.")
            self.send_emisor.set()

        
        self.tiempo_learn_queue = self.tiempo_learn_queue / 1e9  # Convertir a segundos.

        self.tiempo_final_total = time.perf_counter() - tiempo_inicio_total  # Calcular el tiempo total de ejecución.
        
        
        self.fin_proceso.set()
        self.fin_proceso_emisor.set()
        self.proceso_receptor.join()
        if self.proceso_receptor.is_alive():
            print(f"El hilo RECEPTOR no ha terminado. Nodo: {self.id}.")

        self.proceso_emisor.join()        
        if self.proceso_emisor.is_alive():
            print(f"El hilo EMISOR ha terminado. Nodo: {self.id}.")
        
        self.tam_conj_prot = self.diezmar()  # Diezmar la lista de tamaños de conjuntos de prototipos.
        self.tam_lotes_recibidos = self.tam_lotes_recibidos.get_list(0)  # Obtener la lista de tamaños de lotes recibidos.
        
        self.compartidos_final = self.compartidos.pop(0) # Obtener prototipos compartidos.
        self.no_comp_jsd_final = self.no_comp_jsd.pop(0)  # Obtener prototipos no compartidos.
        self.shared_times_final = self.shared_times.pop(0)  # Obtener el número de veces que se compartió.
        self.tiempo_share_final = self.tiempo_share.pop(0)  # Obtener el tiempo total de "share".
        self.tiempo_no_share_final = self.tiempo_no_share.pop(0)  # Obtener el tiempo total de "no share".
        self.protos_descartados_final = self.protos_descartados.pop(0)  # Obtener el número de prototipos descartados.
        
        self.clust_runs = self.modelo_proto.clust_runs
        self.clust_time = self.modelo_proto.clust_time
            
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
            self.modelo_proto.learn_one(x, y)
                        
        self.tiempo_learn_data += (time.perf_counter() - temp)
                        
            
    def learn_from_queue(self):
        
        self.update_cola_index()
        #Si la cola está vacía, se salta el nodo
        
        # print(f"El nodo {self.id} está procesando la cola {self.cola_index}, que tiene {len(self.cola_protos[self.cola_index])} prototipos.")
        colas_revisadas = 0
        colas_a_revisar = self.nodos - 1
        
        while colas_revisadas < colas_a_revisar:
            
            if self.cola_protos.get_length(self.cola_index, call_method = "LEARNING QUEUE. Getting number of protos of neighbour.") > 0:
                proto = self.cola_protos.getleft(self.cola_index, call_method = "LEARNING QUEUE. Popping proto from neighbour.")
                random_index = random.randint(0, len(proto) - 1)
                proto = proto.pop(random_index) 
                
                self.protos_train += 1
                print(f"PROTO {self.protos_train}, NODO {self.id}\n") if self.protos_train % 10000 == 0 else None
                
                x = {str(indice): valor for indice, valor in enumerate(proto['x'])}
                y = proto['y']
                
                temp = time.perf_counter()
                #TRAIN
                self.modelo_proto.learn_one(x, y)
                
                if not (self.modelo_pred is self.modelo_proto):
                    self.modelo_proto.learn_one(x, y)

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
            
        
    def share(self, id, puerto_base, vecinos, send_emisor, fin_proceso_emisor, last_set, s, T, shared_times, compartidos, tiempo_share, tiempo_no_share, no_comp_jsd):
        
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
            no_comp_jsd_local = 0
            
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
                    print(f"[NODO {id}] . Va a añadir {no_comp_jsd_local} a la lista de no_comp_jsd.")
                    no_comp_jsd.append(0, no_comp_jsd_local)
                    # print(f"[NODO {id}] . Item añadido: {compartidos.get_item(0, 0)}.")
                    tiempo_no_share.append(0, tiempo_no_share_local)
                    print(f"[NODO {id}] El hilo emisor ha terminado. ORDEN {fin_proceso_emisor} / {fin_proceso_emisor.is_set()}. Vuelve al join.")
                    return
                
                elif send_emisor.is_set():
                    
                    send_emisor.clear()
                    inicio_share = time.perf_counter()

                    if random.random() < T:
                        
                        shared_times_local += 1                         
                        # print(f"El nodo {id} ha compartido {shar_prev + 1} veces.") if id == 0 else None
                        # Obtener los prototipos del modelo
                        protos = last_set.getleft(id, call_method = "SHARE. Getting own set for sharing.")

                        # Serializar los datos a compartir
                        proto_to_share = pickle.dumps({"id": id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})

                        # Seleccionar aleatoriamente 's' vecinos
                        vecinos_seleccionados = random.sample(vecinos, s)
                        vecinos_eficientes = self.check_sharing(vecinos_seleccionados, last_set, id)
                        compartidos_local += len(protos) * len(vecinos_eficientes)
                        no_comp_jsd_local += len(protos) * (len(vecinos_seleccionados) - len(vecinos_eficientes))
                        # print(f"El nodo {id} ha compartido {comp_previo + sumando} prototipos.") if id == 0 else None
                        
                        # Enviar los prototipos a los vecinos seleccionados
                        for vecino in vecinos_eficientes:
                            # Preparar el mensaje para enviar a través de ZeroMQ ROUTER
                            client_socket.send_multipart([f"{vecino}".encode(), proto_to_share])

                    
                    tiempo_share_local += time.perf_counter() - inicio_share  # Acumular tiempo en "share".
                    # print(f"Tiempo Acumulado en Share: {previo + t_share}.") if id == 0 else None
        
                else: 
                    time.sleep(0.05)
                    tiempo_no_share_local += time.perf_counter() - inicio_no_share  # Acumular tiempo en "no share".
                
                    
        except Exception as e:
            client_socket.setsockopt(zmq.LINGER, 0)
            client_socket.close()
            client_context.term()
            print(f"[ERROR] en SHARE datos en el nodo {id}: {e}")
            return
            
       
    def check_sharing(self, destinos, last_set, id):
    
        mi_conj = last_set.getleft(id, call_method= "CHECKING SHARE. Getting own set for checking.")
        mi_conj = np.array([np.append(proto['x'], proto['y']) for proto in mi_conj])
        # conj_prot[id].append(mis_protos)

        
        destinos_eficiente = []
        for destino in destinos:
            if last_set.get_length(destino, call_method = "CHECKING SHARE. Getting number of sets of neighbour") < 1:
                destinos_eficiente.append(destino)
                # print(f"[NODO {id}] comparte con el nodo {destino} porque no tiene última versión.") 
                continue
            
            # #Vamos a printear los conjuntos de prototipos a evaluar
            dest_conj = last_set.getleft(destino, call_method = "CHECKING SHARE. Getting neighbour set for checking.")
            # # Vamos a hacer un print mostrando ambos conjuntos:
            # print(f"[NODO {id}] MI conjunto: {len(mi_conj)}.") if id == 0 else None
            # print(f"[NODO {id}] Conjunto NODO {destino}: {len(dest_conj)}.") if id == 0 else None
            distancia = jsd.monte_carlo_jsd(mi_conj, dest_conj)
            # print(f"Distancia entre conjuntos de prototipos del nodo {id} y el nodo {destino}: {distancia}.")  if id == 0 else None
            if distancia > 0.5:
                # print(f"Es eficiente compartirle al nodo {destino}.") if id == 0 else None
                destinos_eficiente.append(destino)
            
            # print(f"Los vecinos eficientes son: {destinos_eficiente}.") if id == 0 else None
                
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


    def recibir(self, cola_protos, nodos, puerto_base, id, s, T, fin_proceso, tam_lotes_recibidos, last_set, protos_descartados):
        
        print(f"[NODO {id}] ha iniciado el hilo receptor.")
        if nodos == 1 or s == 0 or T == 0:
            protos_descartados.append(0, 0)
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
        lista_tam_lotes_recibidos = []
        protos_descartados_local = 0
        while True:
            try:
                # Bloquear hasta que un mensaje esté disponible
                identidad, mensaje = server_socket.recv_multipart()
                # id_emisor = identidad.decode()  # Convertir la identidad del emisor de bytes a str si es necesario
                
                data = pickle.loads(mensaje)
                id_recibido = data["id"]  
                protos = data["protos"]
                # print(f"[NODO {id}] Ha recibido de [NODO {id_recibido}]. len={len(protos)}: {protos}.") if id == 0 else None
                
                # print(f"[NODO {id}] Recibido {len(protos)} prototipos de: NODO {id_recibido}.") if id == 0 else None
                            
                # Procesar los prototipos recibidos
                # Por ejemplo, añadir los prototipos recibidos a la cola correspondiente para su procesamiento
                # print(f"[NODO {id}] Va a añadir conjunto a la cola tocha de protos.") 
                
                
                # Calcular cuantos prototipos había antes, y cuantos después de añadir a la cola
                # Ver así cuantos se han descartado.
                # len(protos a añadir) - (get_length despúes - get_length antes)
                # cola_protos.clear(id_recibido, call_method = "RECEIVING. Clearing neighbour queue.")
                # cola_protos.extendleft(id_recibido, protos, call_method = "RECEIVING. Extending left neighbour queue.")
                
                n_before = cola_protos.get_length(id_recibido, call_method = "RECEIVING. Getting the number of protos of neighbour.")
                protos_descartados_local += n_before
                cola_protos.append(id_recibido, protos, call_method ="RECEIVING. Appending protos to neighbour queue.")

                
                # print(f"[NODO {id}] Ha añadido. Procede a actualizar la lista de conjunto reciente.")
                self.update_conj_proto(id, id_recibido, protos, last_set)
                # print(f"[NODO {id}] Ha actualizado la lista de conjunto reciente.")
                lista_tam_lotes_recibidos.append((id_recibido, len(protos)))

    
            except zmq.Again:
                # print(f"El nodo {id} ha agotado el tiempo de espera.")
                if fin_proceso.is_set():
                    server_socket.setsockopt(zmq.LINGER, 0)
                    server_socket.close()
                    server_context.term()
                    for item in lista_tam_lotes_recibidos:
                        tam_lotes_recibidos.append(0, item)
                        
                    protos_descartados.append(0, protos_descartados_local)
                        
                    print(f"[NODO {id}] ha terminado de recibir. Vuelve al join.")
                    return
                # else:
                #     print(f"El nodo {id} lleva {timeout_s} segundos esperando. Pero no ha acabado de recibir")
            
            except Exception as e:
                server_socket.setsockopt(zmq.LINGER, 0)
                server_socket.close()   
                server_context.term()
                print(f"[ERROR] al recibir datos en el nodo {self.id}: {e}")
                return
        

        
    def update_conj_proto(self, id, id_recibido, protos, last_set):
        protos_transformed = np.array([np.append(proto['x'], proto['y']) for proto in protos])
        # print(f"[NODO {id}] Actualiza last_set[{id_recibido}] con len={len(protos_transformed)}: {protos_transformed}.") 
        last_set.append(id_recibido, protos_transformed, call_method = "UPDATING SET. Updating neighbour set.")
        
        
        
