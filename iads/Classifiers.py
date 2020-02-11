# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import math

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
     #TODO: A Compléter

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.vecteur = []
        for i in range(input_dimension):
            self.vecteur.append(random.random()*2-1)
            
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        compteur = 0
        for i in range(dataset.size()):
            x = dataset.getX(i)
            alpha = x.dot(self.vecteur)
            if((alpha * dataset.getY(i))>=0):
                compteur +=1
                print(compteur*100.0/dataset.size())

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    #TODO: A Compléter
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.vecteur = []
        for i in range(input_dimension):
            self.vecteur.append(random.random()*2-1)
            
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        
        alpha = x.dot(self.vecteur)
        if alpha > 0:
            return 1
        else:
            return -1
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        
# ---------------------------
# class ClassifierKNN(Classifier):
#     """ Classe pour représenter un classifieur par K plus proches voisins.
#         Cette classe hérite de la classe Classifier
#     """

#     #TODO: A Compléter
 
#     def __init__(self, input_dimension, k):
#         """ Constructeur de Classifier
#             Argument:
#                 - intput_dimension (int) : dimension d'entrée des exemples
#                 - k (int) : nombre de voisins à considérer
#             Hypothèse : input_dimension > 0
#         """        
#         self.dimension = input_dimension
#         self.neighbors = k
        
#     def dist_euclidienne(x,y):
#         return math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))
        
#     def predict(self, x):
#         """ rend la prediction sur x (-1 ou +1)
#         """
#         tab_dist = list()#tableau des distances
#         #calcul dist x des points de mon dataset, contien [distance, label]
#         for i in range(self.dataset.size()):
#             tab_dist.append([self.dist_euclidienne(x, self.dataset.getX(i)), self.dataset.getY(i)[0]])
            
        
#         nearest = np.argsort(tab_dist, 0)
        
#         #compter nombre de labels "+1" dans les k premiers
#         cpt = 0
#         for i in range(self.k):
#             index = nearest[i][0]#index de la i eme case la plus proche
#             cpt += tab_dist[index][1] #label
#             #print tab_dist[index]
#             #print cpt
#         if cpt > 0:
#             return +1
#         return -1
    
#     def train(self, labeledSet):
#         """ Permet d'entrainer le modele sur l'ensemble donné
#         """
#         self.set = labeledSet
#         # ---------------------------

#     def accuracy(self, dataset):
#         #pour chaque x, y dans dataset comparer predict(x) et y
#         #predict depend du modèle 
#         cpt = 0;
#         for i in range(dataset.size()):
#             if self.predict(dataset.getX(i)) == dataset.getY(i):
#                 cpt+=1
#         return float(cpt) / dataset.size()
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """        
        self.dimension = input_dimension
        self.neighbors = k
        
    def dist_euclidienne(self,x,y):
        somme = 0
        for i in range(self.dimension):    
            somme += math.pow(x[i] - y[i], 2)
                        
        return math.sqrt(somme)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        tab_dist = list()#tableau des distances
        #calcul dist x des points de mon dataset, contien [distance, label]
        for i in range(self.dataset.size()):
            tab_dist.append([self.dist_euclidienne(x, self.dataset.getX(i)), self.dataset.getY(i)[0]])
            
        
        nearest = np.argsort(tab_dist, 0)
        
        #compter nombre de labels "+1" dans les k premiers
        cpt = 0
        for i in range(self.neighbors):
            index = nearest[i][0]#index de la i eme case la plus proche
            cpt += tab_dist[index][1] #label
            #print tab_dist[index]
            #print cpt
        if cpt > 0:
            return +1
        return -1
    
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.dataset = labeledSet
        # ---------------------------

    def accuracy(self, dataset):
        #pour chaque x, y dans dataset comparer predict(x) et y
        #predict depend du modèle 
        cpt = 0;
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) == dataset.getY(i):
                cpt+=1
        return float(cpt) / dataset.size()

    
class ClassifierMoindreCarres(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.dimension = dimension_kernel
        self.epsilon   = learning_rate
        self.w = np.random.rand(self.dimension)
        self.k = kernel
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        ##TODO
        z = np.dot(self.k.transform(x), self.w)
        return z
    
    def accuracy(self, dataset):
        #pour chaque x, y dans dataset comparer predict(x) et y
        #predict depend du modèle 
        cpt = 0;
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) >= (dataset.getY(i)-0.5) and\
            self.predict(dataset.getX(i)) <= (dataset.getY(i)+0.5):
                cpt+=1
        return float(cpt) / dataset.size()
    """
    
        s=0
        p=0
        i=0
        while(i<dataset.size()):
            s+=(self.predict(dataset.getX(i))-dataset.getY(i))**2
            i=i+1
        p=s/dataset.size()"""
    
    def train(self,labeledSet,iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        for i in range(iteration):
            indice = random.randint(0,labeledSet.size()-1)             
            self.w += 2*self.epsilon*self.k.transform(labeledSet.getX(indice))*\
            (labeledSet.getY(indice)-self.predict(labeledSet.getX(indice)))
            
            
            
            
class ClassifierPerceptronKernelNonStock(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.dimension = dimension_kernel
        self.epsilon   = learning_rate
        self.w = np.random.rand(self.dimension)
        self.k=kernel
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        ##TODO
        z = np.dot(self.k.transform(x), self.w)
        if z > 0:
            return +1
        else:
            return -1
        
    def accuracy(self, dataset):
        #pour chaque x, y dans dataset comparer predict(x) et y
        #predict depend du modèle 
        cpt = 0;
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) == dataset.getY(i):
                cpt+=1
        return float(cpt) / dataset.size()       
    
        
    def train(self,labeledSet,iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ##TODO
        for i in range(iteration):
            for j in range(labeledSet.size()):
                if(self.predict(labeledSet.getX(j))!=labeledSet.getY(j)):
                    self.w += self.epsilon*labeledSet.getY(j)*self.k.transform(labeledSet.getX(j))

     