import numpy as np


def parsear_parametros(mensaje):
    print(f"Se van a parsear los parametros: {mensaje}")
    partes = mensaje.split('_')
    
    dataset = partes[0]
    s_value = int(partes[1][1:])
    t_value = float(partes[2][1:])
    limite = int(partes[3][5:])
    ranges = partes[4].split('-')
    inf_range = float(ranges[0][5:])
    sup_range = float(ranges[1][:])
    target_range = (inf_range, sup_range)
    lim_range_searched = (limite, target_range)
    iteracion = int(partes[5][2:])   
    
    dataset_index = datasets.index(dataset)
    s_index = S.index(s_value)
    t_index = np.where(np.isclose(T, t_value))[0][0]
    target_index = lim_range.index(lim_range_searched)
    
    return (iteracion, dataset_index, s_index, t_index, target_index)


def indices_a_parametros(indices):
    iteration, dataset_index, s_index, t_index, limit_target_index = indices
    dataset = datasets[dataset_index]
    s = S[s_index]
    T_value = T[t_index]
    limit, target_range = lim_range[limit_target_index]
    target_range_str = "-".join(map(str, target_range))
    return f"{dataset}_s{s}_T{T_value}_limit{limit}_range{target_range_str}_it{iteration}"
    



T_MAX_IT = 300  # Tiempo máximo de ejecución del hilo
S = [i for i in range(1, 5)]
T = np.array([i for i in range(0, 1001, 50)])
T = T / 1000
tasa_llegadas = 10
media_llegadas = 1 / tasa_llegadas

iteraciones = 50
datasets = ["elec", "phis", "elec2"]

data_name = {"elec": "electricity.csv", "phis": "phishing.csv", "elec2": "electricity.csv"}

datasets = ["elec"]
S = [1, 4]
T = np.array([0.0, 0.1, 1.0])
lim_range = [
    (50, (50, 60)),
    (150, (50, 60)),
    (250, (50, 60)),
    (500, (72.5, 77.5))
        ] 
        

mensaje = "COMENZAR elec_s1_T0.0_limit500_range72.5-77.5_it0_nodo1"
_, params = mensaje.split(' ', 1)
parsed = parsear_parametros(params)

print(f"INDEXES: {parsed}")

to_params = indices_a_parametros(parsed)

print(f"TO PARAMS AGAIN: {to_params}")