import numpy as np
import networkx as nx
from networkx.algorithms.cycles import find_cycle
import matplotlib.pyplot as plt

# Carga el archivo .npy
loaded_matrix = np.load('learned_adjacency_matrix.npy')
#print(loaded_matrix.shape)
# PARA ASEGURARSE DE QUE NO HAY CICLOS CONSIGO MISMO
diagonal_values = np.diagonal(loaded_matrix)
print(diagonal_values)

# Obtener los índices de los elementos distintos de cero
non_zero_indices = np.argwhere(loaded_matrix != 0)
#print(non_zero_indices.shape)
# Formatear los índices a un formato deseado (en este caso, sin decimales)
formatted_indices = ["Cliente_{} Cliente_{}".format(int(index[0]), int(index[1])) for index in non_zero_indices]

# Guardar los índices en un archivo de texto
with open('adj_relations.txt', 'w') as f:
    for formatted_index in formatted_indices:
        f.write(formatted_index + '\n')

# Guarda la matriz en un archivo de texto
with open('adjacency_matrix.txt', 'w') as f:
    for row in loaded_matrix:
        formatted_row = ' '.join([str(int(value)) for value in row])
        f.write(formatted_row + '\n')

# Crear un grafo vacío
dag = nx.DiGraph()

# Agregar las aristas al grafo a partir de los índices formateados
for formatted_index in formatted_indices:
    nodes = formatted_index.split()  # Divide la cadena en dos nodos
    if len(nodes) == 2:
        dag.add_edge(nodes[0], nodes[1])

# Verificar si el grafo es un DAG (sin ciclos)
if not nx.is_directed_acyclic_graph(dag):
    print("El grafo contiene ciclos, se requiere ajuste.")

# Imprimir las aristas del grafo
# print("Aristas del DAG:")
# print(dag.edges())

# Identificar ciclos en el grafo
#cycles = list(nx.simple_cycles(dag))
cycles = list(find_cycle(dag, orientation="ignore"))

# Mostrar los nodos involucrados en los ciclos (NO APARECEN TODOS, REALMENTE HAY MÁS)
for cycle in cycles:
    print("Ciclo:", cycle)

pos = nx.spring_layout(dag)
nx.draw(dag, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", connectionstyle="arc3,rad=0.1")
plt.title("Ciclos en la Red Bayesiana")
plt.show()