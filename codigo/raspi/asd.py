import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Cargar el archivo .arff
file_path = "airlines.arff"
data, meta = arff.loadarff(file_path)

# Convertir a DataFrame
df = pd.DataFrame(data)

# Decodificar valores binarios y bytes en cadenas
for col in df.columns:
    if df[col].dtype == object:  # Si la columna es de tipo object
        df[col] = df[col].str.decode("utf-8")

# Codificar columnas categóricas
categorical_columns = ["Airline", "AirportFrom", "AirportTo"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalizar las columnas numéricas
numeric_columns = ["Flight", "Time", "Length"]
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Guardar como CSV
output_csv = "airlines_normalized.csv"
df.to_csv(output_csv, index=False)

# Mensaje final
print(f"Dataset normalizado y guardado en '{output_csv}'.")


import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset generado en CSV
csv_path = "airlines_normalized.csv"
df = pd.read_csv(csv_path)
df = df.head(50000)

# Contar las etiquetas
label_counts = df['Delay'].value_counts()

# Mostrar la distribución en consola
print("Distribución de las etiquetas:")
print(label_counts)

# Visualizar la distribución
label_counts.plot(kind='bar', rot=0)
plt.title("Distribución de las etiquetas (Delay)")
plt.xlabel("Etiqueta")
plt.ylabel("Cantidad")
plt.xticks(ticks=[0, 1], labels=["0 (No Delay)", "1 (Delay)"])
plt.show()
