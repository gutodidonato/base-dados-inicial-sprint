import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

"""
    Utilizei o https://transparencia.metrosp.com.br/dataset/pesquisa-origem-e-destino
    é um dataset do metroSP informando a origem, destino de passageiros, população e outros fatores
"""


df = pd.read_csv(
    r"C:\Users\Guto\Downloads\Base de dados\projeto\documents\mobilidade.csv",
    encoding="utf8",
    sep=",",
)
print(df.head(10))

"""
    ONDE EXISTE MAIS SALDOMOVIMENTACAO É A ZONA MAIS IMPORTANTE PARA FOCAR O PROJETO INICIAL
"""

df.info()
df[["Atraidas", "População", "Empregos", "Produzidas", "Particulares"]] = df[
    ["Atraidas", "População", "Empregos", "Produzidas", "Particulares"]
].replace("  -   ", np.nan)
df.dropna(
    subset=["Atraidas", "População", "Empregos", "Produzidas", "Particulares"],
    inplace=True,
)
df[["Atraidas", "População", "Empregos", "Produzidas", "Particulares"]] = df[
    ["Atraidas", "População", "Empregos", "Produzidas", "Particulares"]
].astype(float)

"""MANIPULEI O DF PARA PREPARAR OS VALORES"""

df_escolhido = df[["População", "Particulares", "Produzidas", "Atraidas"]]
print(df_escolhido.head())

"""
    Gerar IA
"""
plt.scatter(df["População"], df["Atraidas"])
plt.show()
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_escolhido)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.title("Metodo do cotovelo")
plt.xlabel("Numero de clusters")
plt.ylabel("Inertia")
plt.show()


"""
    O valor bom para K vai ser 5 
"""
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_escolhido)
