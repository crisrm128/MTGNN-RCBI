import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import numpy as np

file_name = "./data/electricity.txt"

fin = open(file_name)
data = np.loadtxt(fin, delimiter=',')

num_clients = data.shape[1]
client_indices = list(range(num_clients))


num_splits = 5  # Número de divisiones en la validación cruzada
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Iterar sobre las divisiones de la validación cruzada
for fold, (train_val_indices, test_indices) in enumerate(kf.split(client_indices)):
    print(train_val_indices.shape)
    print(test_indices.shape)
    # Dividir los índices de train_val en conjuntos de entrenamiento y validación
    train_indices, val_indices = train_test_split(train_val_indices, train_size=192, test_size=64, random_state=42)
    print(train_indices.shape)
    print(val_indices.shape)

    # Ajustar los porcentajes a 60% entrenamiento, 20% validación, 20% prueba
    train_percentage = 0.6
    val_percentage = 0.2
    test_percentage = 1 - train_percentage - val_percentage

    # Ajustar tamaños de los conjuntos en función de los porcentajes
    num_train = int(train_percentage * num_clients)
    num_val = int(val_percentage * num_clients)
    num_test = num_clients - num_train - num_val

    print(num_train)
    print(num_val)
    print(num_test)

    # Seleccionar las columnas correspondientes a los conjuntos de entrenamiento, validación y prueba
    train_subset = data[:, train_indices[:num_train]]
    val_subset = data[:, val_indices[:num_val]]
    test_subset = data[:, test_indices[:num_test]]

    print(train_subset.shape)
    print(train_indices[:num_train])
    print(val_subset.shape)
    print(val_indices[:num_val])
    print(test_subset.shape)
    print(test_indices[:num_test])

    #print(f'Fold {fold + 1} - Porcentaje de entrenamiento: {train_percentage * 100:.2f}%, Porcentaje de validación: {val_percentage * 100:.2f}%, Porcentaje de test: {test_percentage * 100:.2f}%')

    # ---- ESTA VERSIÓN NO CONTEMPLA LAS DIVISIONES 60-20-20, SINO QUE REALIZA LA DIVISIÓN EN BASE AL NÚMERO DE PLIEGUES ----
    # Seleccionar las columnas correspondientes a los conjuntos de entrenamiento, validación y prueba
    # train_subset = data[:, train_indices]
    # val_subset = data[:, val_indices]
    # test_subset = data[:, test_indices]

    # print(train_subset.shape)
    # print(train_indices)
    # print(val_subset.shape)
    # print(val_indices)
    # print(test_subset.shape)
    # print(test_indices)

    # train_percentage = len(train_indices) / num_clients
    # val_percentage = len(val_indices) / num_clients
    # test_percentage = len(test_indices) / num_clients

    # print(f'Fold {fold + 1} - Porcentaje de entrenamiento: {train_percentage * 100:.2f}%, Porcentaje de validación: {val_percentage * 100:.2f}%, 
    #       Porcentaje de test: {test_percentage * 100:.2f}%')