import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Crear un conjunto de datos más grande para clientes de una tienda de vinos
np.random.seed(42)
num_clients = 200

# Generar datos basados en preferencias de vino para diferentes clusters
data = {
    'Tiempo de Permanencia': np.random.randint(10, 60, num_clients),
    'Cantidad de Productos Vistos': np.random.randint(5, 40, num_clients),
    'Frecuencia de Compra': np.random.randint(1, 5, num_clients),
    'Gasto Total': np.random.randint(50, 500, num_clients),
    'Porcentaje de Alcohol': np.concatenate([
        np.random.uniform(10, 13, num_clients // 4),
        np.random.uniform(12, 15, num_clients // 4),
        np.random.uniform(14, 16, num_clients // 4),
        np.random.uniform(11, 14, num_clients // 4),
    ]),
    'Alcalinidad': np.concatenate([
        np.random.uniform(18, 22, num_clients // 4),
        np.random.uniform(20, 25, num_clients // 4),
        np.random.uniform(24, 28, num_clients // 4),
        np.random.uniform(19, 23, num_clients // 4),
    ]),
    'Precio': np.concatenate([
        np.random.uniform(10, 30, num_clients // 4),
        np.random.uniform(20, 50, num_clients // 4),
        np.random.uniform(40, 80, num_clients // 4),
        np.random.uniform(15, 40, num_clients // 4),
    ])
}

df = pd.DataFrame(data)

# Normalizar los datos (es importante para K-Means)
df_normalized = (df - df.mean()) / df.std()

# Seleccionar el número de clústeres (k)
k = 4

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_normalized)

# Visualizar los resultados en un gráfico de dispersión tridimensional
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Porcentaje de Alcohol'], df['Alcalinidad'], df['Precio'], c=df['Cluster'], cmap='viridis', s=50)

# Configurar etiquetas y título
ax.set_xlabel('Porcentaje de Alcohol')
ax.set_ylabel('Alcalinidad')
ax.set_zlabel('Precio')
ax.set_title('Agrupamiento de Clientes con K-Means (Tienda de Vinos)')

# Configurar leyenda
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)

plt.show()

# Mostrar los resultados
print("Asignación de Clústeres:")
print(df[['Tiempo de Permanencia', 'Cantidad de Productos Vistos', 'Frecuencia de Compra', 'Gasto Total', 'Porcentaje de Alcohol', 'Alcalinidad', 'Precio', 'Cluster']])
