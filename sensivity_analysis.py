import pandas as pd

# Lee los dos archivos CSV
df1 = pd.read_csv('predict_confinterv_result.csv', header=None)
df2 = pd.read_csv('predict_confinterv_result_sensitivity_global.csv', header=None)

# Selecciona la tercera fila de ambos DataFrames
tercera_fila_df1 = df1.iloc[2]
tercera_fila_df2 = df2.iloc[2]

# Calcula la diferencia entre las dos filas
diferencia = abs(tercera_fila_df1 - tercera_fila_df2)

# Muestra el resultado
print("Diferencia entre la tercera fila de archivo1 y archivo2:")
print(diferencia)

mayores_a_10 = diferencia[diferencia > 10]

print("Valores de diferencia mayores a 10:")
print(mayores_a_10)

mayores_a_10.to_csv('diferencias_mayores_a_10.txt', header=False)