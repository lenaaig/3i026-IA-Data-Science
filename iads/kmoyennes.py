# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(df):
    maxtab = df.max()
    mintab = df.min()
    a = df - mintab
    a = a/(maxtab-mintab)
    return a
# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(vect1,vect2):
    res = np.dot(vect1-vect2,vect1-vect2)
    return math.sqrt(res)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(matrice):
    return matrice.mean().to_frame().T


# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    d = centroide(df)
    som = [pow(dist_vect(d.iloc[0],i),2) for index,i in df.iterrows()]
    return sum(som)
# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K, df):
    exemples = df.sample(K)
    return exemples

# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(exemple, centroides):
    proches = [dist_vect(exemple,i) for index,i in centroides.iterrows()]
    return proches.index(min(proches))
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(df,centroides):
    dico = dict()
    i=0
    for index,row in centroides.iterrows():
        dico[i] = list()
        i+=1
    for index,row in df.iterrows():
        dico[plus_proche(row,centroides)].append(index)
    return dico
# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(df,base):
    res = pd.DataFrame()
    for i in base.keys():
        res = res.append(df.iloc[base[i]].mean().to_frame().T)
    return res

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(df,matrix):
    som = 0
    for i in matrix.values():
        som += inertie_cluster(df.iloc[i])
    return som
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(k,dataset,epsilon,iter_max):
    centroides = initialisation(k,dataset)
    cluster = affecte_cluster(dataset,centroides)
    iter_cour = 0
    j_p = inertie_globale(dataset,cluster)
    j_p_1 = 0
    centroides = nouveaux_centroides(dataset,cluster)
    while iter_cour < iter_max:
        cluster = affecte_cluster(dataset,centroides)
        j_p_1 = inertie_globale(dataset,cluster)
        centroides = nouveaux_centroides(dataset,cluster)
        iter_cour+=1
        if abs(j_p_1-j_p) < epsilon:
            break
        
    return centroides,cluster
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(df,centroides,matrice):
    label = df.columns.values
    x = label[0]
    y = label[1]
    for i in matrice.values():
        tmp = df.iloc[i]
        c = np.random.rand(3)
        plt.scatter(tmp[x],tmp[y],color=c)
    plt.scatter(centroides['X'],centroides['Y'],color='r',marker='x')
        # Remarque: pour les couleurs d'affichage des points, quelques exemples:
        # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
        # voir aussi (google): noms des couleurs dans matplolib

