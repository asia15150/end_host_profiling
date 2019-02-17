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
def readAnnotatedTrace():
    #ip = 'srcIp:882'
    #G = Graphlet(ip)
    graphlets_ = {}
    with open('annotated-trace.csv') as csv_file:
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
               #line_count+=1
            #else:
               #break
    
    return graphlets_


def compute_walk(length, matrix, size):
    A = matrix
    I = np.zeros((size, size), int)
    np.fill_diagonal(I, 1)
    #print(A)
    #print(I)
    I = np.add(I, A)
    for i in range(length-1):
        #print("####walk: "+str(i+1))
        #print(A)       
        A = np.matmul(A, matrix)
        #print(A)
        I = np.add(I, A)
    #print("####walk: 4")
    #print(A)
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

    
graphlets_ = readAnnotatedTrace()
features = {}
array_matrix = []
array_labels = []
size_max = 0
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
    array_matrix.append(matrix[1])
    if "normal" in g.anomalie:
        array_labels.append("normal")
    else:
        array_labels.append("anomalie")
    #features.update({g.ip_adress: (np.count_nonzero(matrix[1]),g.anomalie)})

array = []
for m in array_matrix:

    a = np.array(m)
    a = np.resize(a,(size_max,size_max))
    a = np.squeeze(a)
    a = a.flatten()
    array.append(a)

array_labels = np.asarray(array_labels)


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



X_train, X_test, y_train, y_test  =  train_test_split(array, array_labels, test_size=1.0, random_state=54, shuffle=False)
#print(c)

X = array
y = array_labels


clf =  svm.SVC(kernel='poly', degree=size_max ,gamma='auto')
clf.fit(array, array_labels)


y_pred = clf.predict(X_test)
classes = [ "anomalie", "normal"]
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix, without normalization')

#plt.show()
plt.show()
