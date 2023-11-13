import numpy as np
from scipy.stats import bootstrap

file_name = "./data/electricity.txt"

fin = open(file_name)
data = np.loadtxt(fin, delimiter=',')

n_samples = data.shape[0] # Número de muestras en el conjunto de datos

# Número de réplicas Bootstrap a generar
n_replicas = 1000  # Ajustar para la realización de pruebas

bootstrap_statistics = [] # Lista para almacenar las estadísticas de interés

for _ in range(n_replicas):
    # Muestreo aleatorio con reemplazo de las ventanas temporales (permite seleccionar la misma fila varias veces)
    bootstrap_sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrap_sample = data[bootstrap_sample_indices]

    # Calcular una estadística de interés en la muestra Bootstrap (por ejemplo, media) -> Ajustar para la realización de pruebas
    statistic = np.mean(bootstrap_sample, axis=0)
    bootstrap_statistics.append(statistic) # Ahora, 'bootstrap_statistics' contiene las estadísticas de interés para cada réplica

# Calcular intervalo de confianza del 90% para la estadística
confidence_interval = np.percentile(bootstrap_statistics, [5, 95], axis=0)

# Calcular la diferencia entre los límites superior e inferior para cada variable
interval_differences = confidence_interval[1] - confidence_interval[0]
# Agregar la fila de diferencias al final de confidence_interval
confidence_interval = np.vstack([confidence_interval, interval_differences])

count_greater_than_100 = np.sum(interval_differences > 100)
print("Cantidad de variables con diferencia mayor a 100:", count_greater_than_100)

# Calcula el tamaño promedio de los intervalos de confianza (si es elevado significa que hay mucha incertidumbre)
average_interval_size = np.mean(interval_differences)
print("Tamaño promedio de los intervalos de confianza:", average_interval_size)

# Filtrar las diferencias que son menores o iguales a 100
filtered_differences = interval_differences[interval_differences < 100]
# Calcular el tamaño promedio de los intervalos de confianza para las diferencias filtradas
average_interval_size_filtered = np.mean(filtered_differences)
print("Tamaño promedio de los intervalos de confianza menores a 100:", average_interval_size_filtered)

# Calcular la demanda promedio de cada cliente (promedio a lo largo de las filas)
demanda_promedio_por_cliente = np.mean(data, axis=0)
demanda_promedio_por_cliente = np.round(demanda_promedio_por_cliente, 3)

np.set_printoptions(precision=3)
print("Demanda promedio por cliente: ", demanda_promedio_por_cliente)

# Definir el formato para los números (coma flotante con 3 decimales)
output_format = '%.3f'

# Guardar los resultados en un archivo CSV
output_file = "bootstrap_f_result.csv"
np.savetxt(output_file, confidence_interval, delimiter=',', fmt=output_format)