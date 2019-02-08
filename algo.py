#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:13:23 2019

@author: asia
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:05:42 2019

@author: asia
"""

import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np


#examples - exo 3.
#srcIp = ['12.124.65.34' for i in range(0, 4)]
#protocol = [17 for i in range(0, 4)]
#dstIP = ['12.124.65.88' for i in range(0, 4)]
#sPort = [138 for i in range(0, 4)]
#dPort = [2 for i in range(0, 4)] 
#formats = [srcIp, protocol,dstIP, sPort, dPort]

graphletsList = {} #dictionary of form ipadress:graphlet


srcIp = []
protocol = []
dstIP = []
sPort = []
dPort = [] 
anomalies = []



#generating graphlets once the arrays above are filed with values
#def generateGraphlets(nbGraphlets, formats):
#    for i in range(0,nbGraphlets):#nb of graphlets
#        graphlet = nx.Graph()
#        for j in range(0,5):#nb of network flows
#            node = formats[j][i]
#            graphlet.add_node(node)
#            if j != 0:
#                previousNode = formats[j-1][i]
#                graphlet.add_edge(previousNode,node) 
#        if i == 0 or i == 1 or i == 2:
#            nx.draw(graphlet)
#        print(graphlet.nodes)
#        #print(graphlet.edges)
#        graphlets.append(graphlet)
#
#    
##saving the values in arrays : srcIp, protocol, dstIP, sPort, dPort, anomalies
def saveRowInArrays(row):
    #print(row)
    srcIp.append(row[0])
    protocol.append(row[1])
    dstIP.append(row[2])
    sPort.append(row[3])
    dPort.append(row[4])
    anomalies.append(row[5])    

 #we add new nodes to existing graph
def addFutherNodes(row):
    print(row)
    srcIp = row[0]
    graphlet = graphletsList[srcIp]
    #nodes = list (graphlet.nodes)
    #for g in graphletsList:
    print(graphletsList)
    #addEdgesAndNodes(row, graphlet)
    return graphlet

def addEdgesAndNodes(row, graphlet):
    graphlet.add_node(row[1], name='protocol')
    graphlet.add_node(row[2], name='dstIP')
    graphlet.add_node(row[3], name='sPort')
    graphlet.add_node(row[4], name='dPort')
    for i in range(1,5):#nb of network flows
        #graphlet.add_nodes_from([933,79,6,21,80])
        node = row[i]
        previousNode = row[i-1]
        graphlet.add_edge(previousNode,node) 
    print(graphlet)
    return graphlet
    #print(graphlet.edges)
     #print(graphlet.nodes)
    

    
def constructGraph(row):
    srcIp = row[0]
    #if the graphlet with ip in question exist already ->
    if srcIp in graphletsList:
        print("===== " + srcIp)
        graphlet = addFutherNodes(row)
        return graphlet
    else:
        graphlet = nx.DiGraph()
        graphlet.add_node(row[0], name='srcIp')
        graphlet = addEdgesAndNodes(row, graphlet)
        graphletsList[srcIp] = list(graphlet)
        return graphlet

 
        



def readAnnotatedTrace():
    with open('annotated-trace.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < 29:
                #if line_count == 23 or line_count == 24:
                #graphlet = constructGraph(row)
                    #print(graphlet)
#                    A0 = calculeMatrixA0(graphlet)
#                    A = countPathsOfLength1(graphlet)
#                    A2 = walks2(A)
#                    A3 = walks3(A2, A)
#                    A4 = walks4(A3, A)
#                    sommeWalks(A0,A,A2,A3,A4)
                constructGraph(row)
                line_count += 1
            else:
                line_count += 1


def draw(tab):
    for t in tab:
        graphlet1 = nx.DiGraph()
        graphlet1.add_nodes_from(tab)
        pos = nx.spring_layout(graphlet1)
        nx.draw_networkx_nodes(graphlet1, pos,nodelist=t,node_color='r',node_size=10,alpha=0.8)
    plt.axis('off')
    plt.show()
    
    
def calculeMatrixA0(graph):
    matrix = []
    nodes = graph.nodes
    for rowI in nodes:
        row = []
        for colI in nodes:
            if rowI == colI:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    print("A0 : ")
    print(np.array(matrix))
    return np.array(matrix)
        

def orderingNodes(nodesList):
    #print(nodesList)
    nodesFinales = []
    for node,label in nodesList:
        if (label == 'srcIp'):
            nodesFinales.append(node)
            break
    for node,label in nodesList:
        #print(node + "   " + label)
        if (label == 'protocol'):
            nodesFinales.append(node)
            break
    for node,label in nodesList:
        if (label == 'dstIP'):
            nodesFinales.append(node)
            break
    for node,label in nodesList:
        if (label == 'sPort'):
            nodesFinales.append(node)
            break
    for node,label in nodesList:
        if (label == 'dPort'):
            nodesFinales.append(node)
            break
    print(nodesFinales)
    return nodesFinales



    
#adjacency matrix with walks of lengths 1  
def countPathsOfLength1(graph):
    matrix = []
    edges = graph.edges
    nodes = graph.nodes
    nbNodes = len(nodes)
    nodesList = list(graph.nodes(data='name'))
    nodesFinales = orderingNodes(nodesList)
    print(edges)
    #to fill the matrix row by row
    for node in nodesFinales:
        row = []
        for colI in nodesFinales:
            try:
                #edge in which the current node is linked with another one
                currentEdge =list(edges(node))
                # the node with which the current node is linked
                neighboureEdge = currentEdge[0][1]
                #print("currentEdge " + str(currentEdge))
                #print("neighboureEdge " + str(neighboureEdge))
                # if the colon's label in matrix in which we are right now equals neighbour node of "node"
                if neighboureEdge == colI:
                    row.append(1)
                else :
                    row.append(0)
            except IndexError:
                row.append(0)
        matrix.append(row)
    print(np.array(matrix))
    return np.array(matrix)
  
    
def multiplicationMatrix(A,B):
    return np.matmul(A, B)

def walks2(A):
    res = multiplicationMatrix(A,A)
    print("walk2 : ")
    print(res)
    return res
    
def walks3(A2,A):
    res = multiplicationMatrix(A2,A)
    print("walk3 : ")
    print(res)
    return res

    
def walks4(A3, A):
    res = multiplicationMatrix(A3,A)
    print("walk4 : ")
    print(res)
    return res

def sommeWalks(A0,A, A2,A3,A4):
    res = A0 + A + A2 + A3 + A4
    print("somme : ")
    print(res)
    return res


readAnnotatedTrace()



