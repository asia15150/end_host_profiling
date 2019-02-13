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
import random
from sklearn import svm

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
        self.anomalies = set()
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
        row[5] = 'anomalies:'+row[5]
        
        #self.srcIp.append(row[0])
        self.protocol.add(row[1])
        self.dstIp.add(row[2])
        self.sPort.add(row[3])
        self.dPort.add(row[4])
        self.anomalies.add(row[5])
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


    # draw version 1
    def draw(self):
        #nodes
        self.graph.add_node(self.ip_adress, label = 'srcIp')
        self.graph.add_nodes_from(list(self.protocol), label = 'protocol')
        self.graph.add_nodes_from(list(self.dstIp), label = 'dstIP')
        self.graph.add_nodes_from(list(self.sPort), label = 'sPort')
        self.graph.add_nodes_from(list(self.dPort), label = 'dPort')
        self.graph.add_nodes_from(list(self.anomalies), label = 'anomalies')

        #edges
        self.graph.add_edges_from(self.tab_nodes)
        
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
               
            #if line_count < 1000:#####CHANGE TO 10000 TO SEE DIFFERENC####FOR NOW I ONLY WORK WITH srcIp=882
               #line_count+=1
            #else:
               #break
    
    return graphlets_


def compute_walk(length, matrix):
    A = matrix
    for i in range(length-1):
        #print("####walk: "+str(i+1))
        #print(A)
        A = np.matmul(A, matrix)
    #print("####walk: 4")
    #print(A)
    return A

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


X = []
Y = []
i=0

def before_SVM(B):
    #B.flat
    if (B.size != 900):
        B.resize((1,900), refcheck=False)
        #B.flatten()
        X.append(B[0])
        print(B)
    else:
        X.append(B.flatten())
    print(B.ndim)
    print(B.size)
    Y.append(random.randint(0, 1))


def SVM():
    clf = svm.SVC(gamma='scale')
    c = clf.fit(X, Y)


graphlets_ = readAnnotatedTrace()
for index, g in enumerate(graphlets_.values()):
    if (i == 0 or  i == 1 or i == 2 or i == 3 ):
        #i = 200+index+1
        fig = draw(g)
        
        B = compute_walk(4, g.get_first_matrix())
        draw2(fig, B)

        before_SVM(B)

        
        i+=1


print("****")

#SVM()




    
    #B = compute_walk(4, g.get_first_matrix())
    #df = pd.DataFrame(B, columns=g.graph.nodes(), index=g.graph.nodes())
    #print(df)
    #print(B)
    #print(df)
    #plt.figure(index)

#plt.show()



