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
    tmax=df.max()
    tmin=df.min()
    newm=df-tmin
    newm=newm/(tmax-tmin)
    return newm

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
from math import sqrt
def dist_vect(v1, v2):
    return sqrt(np.dot(v1-v2, v1-v2))

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    return df.mean()

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def fonction_caracteristique_de_E(E, x):
    if (x in E.values):
        return 1
    return 0

def inertie_cluster(df):
    somme=0
    for i in df.values:
        somme+=fonction_caracteristique_de_E(df,i)*(dist_vect(i, centroide(df)))**2
    return somme
# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,df):
    return df.sample(K)
# -------
# ************************* Recopier ici la fonction plus_proche()

def plus_proche(exemple, df):
    min=dist_vect(exemple,df.values[0])
    indi=0
    
    for i in range(1,len(df)):
        res = dist_vect(exemple,df.values[i])
        if(min > res and res != 0):
            min=res
            indi=i
    return indi

# -------
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
def kmoyennes(k,df,e,iter_max):
    centros = initialisation(k,df)
    prems = True
    i = 0
    while(i < iter_max):
        dict_aff = affecte_cluster(df,centros)
        centros = nouveaux_centroides(df,dict_aff)
        if(not prems and abs(inertie_globale(df,dict_aff) - aux) < e):
            return centros,dict_aff
        aux = inertie_globale(df,dict_aff)
        prems = False
        i += 1
    return centros,dict_aff
    
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
    plt.scatter(les_centres['X'],les_centres['Y'],color='r',marker='x')
# -------