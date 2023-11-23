import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Crear un conjunto de datos más grande para clientes de una tienda de vinos
np.random.seed(42)
num_clients = 200
data = {
    'Tiempo de Permanencia': np.random.randint(10, 60, num_clients),
    'Cantidad de Productos Vistos': np.random.randint(5, 40, num_clients),
    'Frecuencia de Compra': np.random.randint(1, 5, num_clients),
    'Gasto Total': np.random.randint(50, 500, num_clients)
}

df = pd.DataFrame(data)

# Normalizar los datos (es importante para K-Means)
df_normalized = (df - df.mean()) / df.std()

# Seleccionar el número de clústeres (k)
k = 4

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_normalized)

# Visualizar los resultados
plt.scatter(df['Tiempo de Permanencia'], df['Gasto Total'], c=df['Cluster'], cmap='viridis')
plt.title('Agrupamiento de Clientes con K-Means (Tienda de Vinos)')
plt.xlabel('Tiempo de Permanencia')
plt.ylabel('Gasto Total')
plt.show()

# Mostrar los resultados
print("Asignación de Clústeres:")
print(df[['Tiempo de Permanencia', 'Cantidad de Productos Vistos', 'Frecuencia de Compra', 'Gasto Total', 'Cluster']])
