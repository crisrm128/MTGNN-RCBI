import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import K2Score
from collections import deque

file_name = "./data/electricity.txt"

# fin = open(file_name)
# data = np.loadtxt(fin, delimiter=',')
data = pd.read_csv(file_name, delimiter=',', header=None)

# Renombrar las columnas
new_column_names = [f"Cliente_{i + 1}" for i in range(data.shape[1])]
data.columns = new_column_names

edges = []

# Lee el archivo "adj_relations.txt" y guarda las cadenas en la lista 'edges' como tuplas de dos elementos
with open('adj_relations.txt', 'r') as f:
    for line in f:
        line = line.strip()  # Elimina caracteres de nueva línea u otros espacios en blanco
        items = line.split()  # Divide la línea en palabras separadas por espacio
        if len(items) == 2:
            edges.append((items[0], items[1]))

print(edges[0])


# ---- INTENTO DE ELIMINAR LAS ARISTAS MANUALMENTE PERO NO ES BUENA PRÁCTICA ----
# Lista de edges a eliminar
# edges_a_eliminar = [('Cliente_11', 'Cliente_4'), ('Cliente_27', 'Cliente_6'), ('Cliente_29', 'Cliente_12')]

# # Itera sobre los edges a eliminar y quítalos de la lista "edges"
# for edge_to_remove in edges_a_eliminar:
#     if edge_to_remove in edges:
#         edges.remove(edge_to_remove)


# Crear un objeto BayesianNetwork
model = BayesianNetwork()

# Agregar las aristas al grafo de la red bayesiana
for edge in edges:
    model.add_edge(*edge)  # Agregar la arista usando la tupla


# Definir las variables (nodos) y sus valores posibles
variables = []

for i in range(data.shape[1]):
    variable_name = f"Cliente_{i + 1}"
    #state_names = [f"Estado_{j}" for j in range(data.shape[0])]
    #variables.append((variable_name, state_names)) #Cada nodo es un cliente con sus valores de demanda asociados
    model.add_node(variable_name)

# # Estimar las probabilidades condicionales a partir de tus datos
# # Usaremos Maximum Likelihood Estimator en este ejemplo
for i in range(data.shape[1]):
    variable_name = f"Cliente_{i + 1}"
    print(f"Estimando tabla de probabilidad condicional para {variable_name}")
    #state_names = [f"Estado_{j}" for j in range(data.shape[0])]
    # state_counts = [data[i][j] for j in range(data.shape[0])]
    cpd = MaximumLikelihoodEstimator(model, data).estimate_cpd(variable_name)
    # cpd.values = state_counts
    model.add_cpds(cpd)

# ---- DEMASIADO COSTOSO COMPUTACIONALMENTE DEBIDO A LA GRAN CANTIDAD DE NODOS ----
# Aprende la estructura de la red Bayesiana usando el algoritmo K2
# hc = HillClimbSearch(data)
# best_model = hc.estimate(scoring_method=K2Score(data))
# Muestra la estructura aprendida
# print(best_model.edges())

# Dibuja el grafo
pos = nx.spring_layout(model)
nx.draw(model, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
edge_labels = {edge: edge[1] for edge in model.edges()}
nx.draw_networkx_edge_labels(model, pos, edge_labels=edge_labels)
plt.title("Grafo de la Red Bayesiana")
plt.show()


# Comprobar si la red bayesiana es válida
assert model.check_model()

# # Aprender la estructura y los parámetros de la red
# model.fit(data)

# Puedes realizar inferencia o imprimir información sobre la red bayesiana
# print("Estructura de la red bayesiana:")
# print(model.get_cpds())
#print(model.get_independencies())