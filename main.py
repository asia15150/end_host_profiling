#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:05:42 2019

@author: asia
"""
#import sys
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


#import pprint

#font = {'size'   : 30}

#plt.rc('font', **font)

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
           # if 'srcIp:882' == ip:
            #    G.saveRowInArrays(row)
            #print(ip)
            #print(G)
            if G == None:
               G = Graphlet(ip)
               G.saveRowInArrays(row)
               graphlets_.update({ip:G})
            else:
               G.saveRowInArrays(row)
               
            #if line_count < 100:#####CHANGE TO 10000 TO SEE DIFFERENC####FOR NOW I ONLY WORK WITH srcIp=882
             #  line_count+=1
            #else:
             #  break
    
    return graphlets_


def compute_walk(length, matrix, size):#compute walk
    A = matrix
    I = np.zeros((size, size), int)
    np.fill_diagonal(I, 1)#matrix identity
    I = np.add(I, A)
    for i in range(length-1):     
        A = np.matmul(A, matrix)
        I = np.add(I, A)
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

def damping_factor(direct_product_matrix):
    #print(direct_product_matrix)
    maxInDegree = 0
    maxOutDegree = 0
    sum_all_columns = direct_product_matrix.sum(axis=0)
    sum_all_row = direct_product_matrix.sum(axis=1)
    l = len(sum_all_columns)
    for index in range(l):
        maxInDegree = max(maxInDegree, sum_all_columns[index])
        maxOutDegree = max(maxOutDegree, sum_all_row[index])
    #print(maxInDegree)
    #print(maxOutDegree)
    
    factor = 1/min(maxInDegree,maxOutDegree)
    return factor

def direct_product_kernel(matrix_adj_A, matrix_adj_B):
    l_A = len(matrix_adj_A)#size of matrix adjacency A
    l_B = len(matrix_adj_B)#size of matrix adjacency B
    
    products = None

    ######the result of [A, A] direct product [A, A] is
    #[AA, AA, AA, AA]
    for i in range(l_A):#for each column of A
        matrix = None
        for j in range(l_A):#for each row of A
            #multiply an element of A with the whole matrix of B
            new_matrix = matrix_adj_A[i][j] * matrix_adj_B #multiply an element of 

            #we concatenate the matrix
            if j == 0:# we initialise new line with the first chunk
                matrix = new_matrix
            else:
                matrix = np.concatenate((matrix, new_matrix), axis=1)
                #we concatenate the next chunk to the same line
        if i == 0:#concatenate by column
            products = matrix
            #print(products)
        else:
            products = np.concatenate((products, matrix), axis=0)
            #print(matrix)

    #we now can compute the goemetric sum
    #(I - dpf*direct_product_matrix)exposant(-1))
    dpf = damping_factor(products)
    
    I = np.identity(len(products))#matrix identity
    
    k = np.linalg.inv(np.subtract(I, dpf*products))#inversion
    return np.array(k).sum()
            
            
                
    
graphlets_ = readTrace('annotated-trace.csv')
features = {}
array_matrix = []
array_labels = []
size_max = 0
adjacencies = []
for index, g in enumerate(graphlets_.values()):
    g.make_graph()
    g.make_first_matrix()
    #i = 200+index+1
    #print(i)
    #plt.subplot(i)
    
    #fig = plt.figure(figsize=(20,20))
    #plt.subplot(2, 1, 1)
    #g.draw()

    #plt.subplot(2, 1, 2)
    #g.make_first_matrix()
    
    #B = compute_walk(4, g.get_first_matrix(), size)
    #df = pd.DataFrame(B, columns=g.graph.nodes(), index=g.graph.nodes())
    #df[0].plot()
    #A = np.squeeze(np.asarray(B))
    #plt.table(cellText=df.values,
    #rowLabels=df.index, colLabels=df.columns,
    #loc='center', fontsize=30)
    #plt.axis('off')
    #df.plot()
    
    #fig.savefig('plots/'+g.ip_adress+'.png', dpi=200)
    #plt.close(fig) 
    
    
    #B = compute_walk(4, g.get_first_matrix())
    #df = pd.DataFrame(B, columns=g.graph.nodes(), index=g.graph.nodes())
    #print(df)
    #print(B)
    #print(df)
    #plt.figure(index)
    size = g.get_labels_size()
    if size > size_max:
        size_max = sizesize = g.get_labels_size()
    matrix = compute_walk(4, g.get_first_matrix(), size)
    array_matrix.append(matrix)
    adjacencies.append(np.array(g.get_first_matrix()))
    if(g.anomalie != -1):
        if "normal" in g.anomalie:
            array_labels.append("normal")
        else:
            array_labels.append("anomalie")
    #features.update({g.ip_adress: (np.count_nonzero(matrix[1]),g.anomalie)})

def reshape_matrix(array_matrix):
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

def draw(g):
    fig = plt.figure(figsize=(20,20))
    plt.subplot(2, 1, 1)
    g.draw()
    plt.subplot(2, 1, 2)
    g.make_first_matrix()
    return fig

def draw2(fig, B):
    df = pd.DataFrame(B, columns=g.graph.nodes(), index=g.graph.nodes())
    A = np.squeeze(np.asarray(B))
    plt.table(cellText=df.values,
                rowLabels=df.index, colLabels=df.columns,
                loc='center', fontsize=30)
    plt.axis('off')
    fig.savefig('plots/'+g.ip_adress+'.png', dpi=200)
    plt.close(fig)


#dpk = direct_product_kernel(adjacencies[0],adjacencies[1])
#l = len(adjacencies) 
#kernel_matrix = np.zeros((l,l))
#print(kernel_matrix.shape)
#time_start = time.process_time()
#run your code

def classification_annoted(array, array_labels, size_max):
    #print(len(array))
    X_train, X_test, y_train, y_test  =  train_test_split(array, array_labels, test_size=1.0, random_state=54, shuffle=False)
    clf =  svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
    clf.fit(array, array_labels)
    y_pred = clf.predict(X_test)
    classes = [ "anomalie", "normal"]
    print('accuracy score: ',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix, without normalization')
    plt.show()

def classification_anannoted(array, array_labels, size_max):
    #print(len(array))
    X_train, X_test, y_train, y_test  =  train_test_split(array, array_labels, test_size=1.0, random_state=54, shuffle=False)
    clf =  svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
    clf.fit(array, array_labels)
    y_pred = clf.predict(X_test)
    classes = [ "anomalie", "normal"]
    print('accuracy score: ',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix, without normalization')
    plt.show()


array_labels = np.asarray(array_labels)
array = reshape_matrix(array_matrix)
classification(array, array_labels, size_max)
#adj = adjacencies
#print(adjacencies)
#adjacencies 
#adjacencies = np.squeeze(adjacencies)


#print(adjacencies)
print("hello")
#clf =  svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
#clf.fit(kernel_matrix)
