import simpy
import random
from collections import deque
import numpy as np

N = 5
T = 1.0  
lambda_muestra = 5  
media_entre_llegadas = 1 / lambda_muestra
s = 4  
L = 250  
TASA_LLEGADA = lambda_muestra * L * T * s / (N - 1)  
t_llegadas = 1 / TASA_LLEGADA  
N_MUESTRAS = 1000
TIEMPO_SIMULACION = media_entre_llegadas * N_MUESTRAS  
TASA_SERVICIO = 300  
MAX_TAMAÑO_COLA = 255000 



class SistemaColas:
    def __init__(self, env, max_tamaño_cola):
        self.env = env
        self.max_tamaño_cola = max_tamaño_cola
        self.colas = [deque(maxlen=max_tamaño_cola) for _ in range(N-1)]
        self.cola_prioridad = deque()
        self.bloqueos = [0 for _ in range(N-1)]
        self.intentos = [0 for _ in range(N-1)]
        self.prototipos_procesados = [0 for _ in range(N)]  # Incluye cola con prioridad como último elemento
        self.evento_completado = env.event()

    def servidor(self):
        last_index = 0
        while True:
            prototipo_actual = None
            cola_origen = None

            # Verificar la cola con prioridad primero
            if self.cola_prioridad:
                prototipo_actual = self.cola_prioridad.pop()
                cola_origen = N-1  # Índice de la cola con prioridad

                        # Si la cola con prioridad está vacía, revisar las colas LIFO
            if prototipo_actual is None:
                for offset in range(N-1):
                    i = (last_index + offset) % (N-1)  # Calcula el índice actual basado en last_index
                    if self.colas[i]:
                        prototipo_actual = self.colas[i].popleft()
                        cola_origen = i
                        last_index = (i + 1) % (N-1)  # Actualiza last_index para la próxima iteración
                        break

            if prototipo_actual:
                # Procesar el prototipo
                yield self.env.timeout(1.0 / TASA_SERVICIO)
                self.prototipos_procesados[cola_origen] += 1  # Contabilizar prototipo procesada
                
                if self.prototipos_procesados[-1] >= 1000:
                    self.evento_completado.succeed()
            else:
                yield self.env.timeout(0.01)  # Espera breve si no hay prototipos

def llegada_prototipos(env, sistema_colas):
    for i in range(N-1):
        env.process(llegada_prototipos_cola(env, sistema_colas, i))
    env.process(llegada_prototipos_cola_prioridad(env, sistema_colas))
    yield env.timeout(0)

def llegada_prototipos_cola(env, sistema_colas, numero_cola):
    while True:
        yield env.timeout(np.random.exponential(t_llegadas))
        sistema_colas.intentos[numero_cola] += 1
        if len(sistema_colas.colas[numero_cola]) < MAX_TAMAÑO_COLA:
            sistema_colas.colas[numero_cola].appendleft(f"prototipo en cola LIFO {numero_cola}")
        else:
            sistema_colas.bloqueos[numero_cola] += 1

def llegada_prototipos_cola_prioridad(env, sistema_colas):
    while True:
        yield env.timeout(np.random.exponential(media_entre_llegadas))
        sistema_colas.cola_prioridad.appendleft("prototipo en cola con prioridad")

# Configurar y ejecutar la simulación
env = simpy.Environment()
evento_fin = env.event()
sistema_colas = SistemaColas(env, MAX_TAMAÑO_COLA)
env.process(llegada_prototipos(env, sistema_colas))
env.process(sistema_colas.servidor())
env.run(until=sistema_colas.evento_completado)

# Estadísticas de bloqueo
total_intentos = sum(sistema_colas.intentos)
total_bloqueos = sum(sistema_colas.bloqueos)
probabilidad_bloqueo_total = total_bloqueos / total_intentos if total_intentos > 0 else 0
for i in range(N-1):
    probabilidad_bloqueo_cola = sistema_colas.bloqueos[i] / sistema_colas.intentos[i] if sistema_colas.intentos[i] > 0 else 0
    print(f"Probabilidad de bloqueo en cola LIFO {i}: {probabilidad_bloqueo_cola}")
print(f"Probabilidad de bloqueo total: {probabilidad_bloqueo_total}")


# Mostrar número de prototipos procesadas y tasas de procesamiento
print("\nNúmero de prototipos procesadas:")
print(f"Cola con prioridad procesó {sistema_colas.prototipos_procesados[N-1]} muestras.")
for i in range(N-1):
    print(f"Cola LIFO {i} procesó {sistema_colas.prototipos_procesados[i]} prototipos.")

# Tasas de procesamiento reales
print(f"\nTasas de procesamiento:")
tasas_procesamiento = [prototipos / env.now for prototipos in sistema_colas.prototipos_procesados]
for i, tasa in enumerate(tasas_procesamiento[:-1]):
    print(f"Tasa de procesamiento real para cola LIFO {i}: {tasa:.2f} prototipos/unidad de tiempo.")
print(f"Tasa de procesamiento real para cola con prioridad: {tasas_procesamiento[-1]:.2f} muestras/unidad de tiempo.")

#Número de prototipos totales y tasas de procesamiento totales
# Imprimir el tiempo total de la simulación
print(f"Tiempo total de la simulación: {env.now}")
print(f"\nNúmero total de prototipos+muestras procesados: {sum(sistema_colas.prototipos_procesados)}")
print(f"Tasa de procesamiento total: {sum(tasas_procesamiento):.2f} prototipos(y muestras)/unidad de tiempo.")
