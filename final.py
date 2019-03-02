#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Wed Jan 30 16:05:42 2019
    
    @author: asia
    """
#import sys
from multiprocessing import Pool

import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt
#from random import choice
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
from mpl_toolkits import mplot3d
import time
from joblib import Parallel, delayed


graphlets_not = {}
graphlets_ = {}

#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(threshold=np.nan)
class Graphlet:
    def __init__(self, ip_adress):
        self.ip_adress = ip_adress
        
        #set is to prevent redundancy
        self.protocol = set()
        self.dstIp = set()
        self.sPort = set()
        self.dPort = set()
        self.anomalies = -1
        self.tab_nodes = set()
        self.graph = nx.DiGraph()
    
    #make edges exemple: [a,b,c] => [(a,b),(b,c),(b,c)]
    def makeNodes(self, row):
        G = nx.path_graph(row)
        return set(list(G.edges)) #return set just to be unioned later on
    
    #parse row
    def saveRowInArrays(self, row):
        #print(row)
        row[0] = 'srcIp:'+row[0]
        row[1] = 'protocol:'+row[1]
        row[2] = 'dstIP:'+row[2]
        row[3] = 'sPort:'+row[3]
        row[4] = 'dPort:'+row[4]
        if len(row) == 6:
            self.anomalie = row[5]
        #row[5] = 'anomalies:'+row[5]
        #self.srcIp.append(row[0])
        self.protocol.add(row[1])
        self.dstIp.add(row[2])
        self.sPort.add(row[3])
        self.dPort.add(row[4])
        #self.anomalies.add(row[5])
        row = row[:-1]
        
        self.tab_nodes = self.tab_nodes.union(self.makeNodes(row))

#build the matrix to do the random walk
    def make_first_matrix(self):
        self.first_matrix = nx.adjacency_matrix(self.graph).todense()

    #get first matrix
    def get_first_matrix(self):
        return self.first_matrix
    
    #concat all the array
    def node_list(self):
        return self.protocol.union(self.dstIp.union(self.sPort.union(self.anomalies))).add(self.ip_adress)
    
    #draw version 2
    def draw_v2(self):
        from_ = []
        to_ = []
        
        for pair in self.tab_nodes:
            (key, value) = pair
            from_.append(key)
            to_.append(value)
        
        df = pd.DataFrame({ 'from':from_, 'to':to_})
        G=nx.from_pandas_edgelist(df, 'from', 'to')
        nx.draw(G, with_labels=True)
    
    def make_graph(self):
        #nodes
        self.graph.add_node(self.ip_adress, label = 'srcIp')
        self.graph.add_nodes_from(list(self.protocol), label = 'protocol')
        self.graph.add_nodes_from(list(self.dstIp), label = 'dstIP')
        self.graph.add_nodes_from(list(self.sPort), label = 'sPort')
        self.graph.add_nodes_from(list(self.dPort), label = 'dPort')
        #self.graph.add_nodes_from(list(self.anomalies), label = 'anomalies')
        
        #edges
        self.graph.add_edges_from(self.tab_nodes)
    
    
    def get_labels_size(self):
        return len(self.graph)
    # draw version 1
    def draw(self):
        #graphlet1.nodes(data=True)
        
        color_map = []
        for node in self.graph:
            if 'srcIp' in node:
                color_map.append('b')
            elif 'protocol' in node:
                color_map.append('r')
            elif 'dstIP' in node:
                color_map.append('green')
            elif 'sPort' in node:
                color_map.append('deepskyblue')
            elif 'dPort' in node:
                color_map.append('yellow')
            else:
                color_map.append('pink')
        nx.draw_shell(self.graph, with_labels=True,node_color = color_map, node_size=500)


#read file and build graphlets
def readTrace(file):
    #ip = 'srcIp:882'
    #G = Graphlet(ip)
    graphlets_ = {}
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            ip = 'srcIp:'+row[0]
            G = graphlets_.get(ip)
            #            if 'srcIp:882' == ip:
            #                print(ip)
            #                print(G)
            if G == None:
                G = Graphlet(ip)
                G.saveRowInArrays(row)
                graphlets_.update({ip:G})
            else:
                G.saveRowInArrays(row)

#            if line_count < 3000:#####CHANGE TO 10000 TO SEE DIFFERENC####FOR NOW I ONLY WORK WITH srcIp=882
#               line_count+=1
#            else:
#               break

    return graphlets_


#the second value returned by this function correspond to random walk kernel
def compute_walk(length, matrix, size):#compute walk
    A = matrix
    I = np.zeros((size, size), int)
    np.fill_diagonal(I, 1)#matrix identity
    #print("\n IDENTITY MATRIX : \n")
    #print(I)
    I = np.add(I, A)
    for i in range(length-1):
        #print("\n RANDOM WALK MATRIX OF LENGTH ", i+1,":\n\n", A)
        A = np.matmul(A, matrix)
        I = np.add(I, A)
    #print("\n RANDOM WALK MATRIX OF LENGTH 4",":\n\n", A)
    #print("A ", A, " I \n", I)
    return [A, I]



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def reshape_matrix(array_matrix, size_max):
    array = []
    for m in array_matrix:
        #print(np.array(array_matrix).shape)
        a = np.array(m)
        #print(a)
        a = np.resize(a,(size_max,size_max))
        a = np.squeeze(a)
        a = a.flatten()
        #print(a)
        array.append(a)
    return array



####à faire!!!!!!!!!donne les matrix non annotated dans cette fonction et
###à la place de y_pred = clf.predict(X_test) il faut faire clf.predict(Matrix_non_annotated )
def classification_annotated_rbf(array, array_labels, size_max, array_not):
    #print(len(array))
    X_train, X_test, y_train, y_test  =  train_test_split(array, array_labels, test_size=0.8, random_state=54, shuffle=False)
    clf =  svm.SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
    
    clf.fit(array, array_labels)
    y_pred = clf.predict(X_test)
    not_pred = clf.predict(array_not)#not annotated prediction
    
    classes = [ "anomalie", "normal"]
    
    
    print('accuracy score: ',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))
    
    print('accuracy score not: ',accuracy_score(y_test, not_pred))
    print(classification_report(y_test, not_pred, target_names=classes))
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix_not = confusion_matrix(y_test, not_pred)
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix for rbf kernel, without normalization')
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix_not, classes=classes,title='Confusion matrix for rbf kernel, annotated matrix with not annotated')
    
    plt.show()




def classification_annotated_linear(array, array_labels, size_max, array_not):
    #print(len(array))
    X_train, X_test, y_train, y_test  =  train_test_split(array, array_labels, test_size=0.8, random_state=54, shuffle=False)
    
    #decomment choose one kernel to see difference
    clf =  svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
    #clf =  svm.SVC(kernel='linear', gamma='auto')
    #clf =  svm.SVC(kernel='poly', degree=size_max, gamma='auto')
    
    clf.fit(array, array_labels)
    
    y_pred = clf.predict(X_test)#test
    not_pred = clf.predict(array_not)#not annotated prediction
    
    classes = [ "anomalie", "normal"]
    
    
    print('accuracy score: ',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))

    print('accuracy score not: ',accuracy_score(y_test, not_pred))
    print(classification_report(y_test, not_pred, target_names=classes))
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix_not = confusion_matrix(y_test, not_pred)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix for linear kernel, annotated matrix with y_test')
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix_not, classes=classes,title='Confusion matrix for linear kernel, annotated matrix with not annotated')
    
    plt.show()


def main_random_walk():
    features = {}
    array_matrix = []#annotated
    array_matrix_not = []#not annotated
    array_labels = []
    size_max = 0
    adjacencies = []
    
    for index, g in enumerate(graphlets_not.values()):
        g.make_graph()
        g.make_first_matrix()
        size = g.get_labels_size()
        matrix = compute_walk(4, g.get_first_matrix(), size)
        array_matrix_not.append(matrix[1])

    for index, g in enumerate(graphlets_.values()):
        #print("***********************")
        g.make_graph()
        g.make_first_matrix()
        #print(g.graph.edges)
        size = g.get_labels_size()
        if size > size_max:
            size_max = sizesize = g.get_labels_size()
        matrix = compute_walk(4, g.get_first_matrix(), size)
        array_matrix.append(matrix[1])


#print("\n RANDOM WALK KERNEL MATRIX : \n")
#print(matrix[1])
#       print('\n')
#draw(g)
# plt.show()

        #adjacencies.append(np.array(g.get_first_matrix()))
        if(g.anomalie != -1):
            if "normal" in g.anomalie:
                array_labels.append("normal")
            else:
                array_labels.append("anomalie")
        #features.update({g.ip_adress: (np.count_nonzero(matrix[1]),g.anomalie)})

    array_labels = np.asarray(array_labels)
    array = reshape_matrix(array_matrix, size_max)
    array_not = reshape_matrix(array_matrix_not, size_max)
    #classification_annotated_rbf(array, array_labels, size_max, array_not)
    classification_annotated_linear(array, array_labels, size_max, array_not)



graphlets_ = readTrace('annotated-trace.csv')
graphlets_not = readTrace('not-annotated-trace.csv')

main_random_walk()



