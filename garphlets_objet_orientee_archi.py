#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np



#dictonary of objects Graphlets
graphletsDict = {}

class Graphlet:
    
    def __init__(self):
        #dictionary of form ipadress:{srcIp : value, protocol : value, dstIP  : value, sPort : value, dPort : values, anomalies :value }
        self.srcIp = -1
        self.protocol = []
        self.dstIP = []
        self.sPort = []
        self.dPort = []
        self.anomalies = -1
        #self.lenLongestList
        self.A0 = np.zeros
        self.A1 = np.zeros
        self.A2 = np.zeros
        self.A3 = np.zeros
        self.A4 = np.zeros
        self.somme = np.zeros

                
    
    def printAllAttributs(self):
        print("self.srcIp : " + str(self.srcIp))
        print("self.protocol : " + str(self.protocol))
        print("self.dstIP : " + str(self.dstIP))
        print("self.sPort: " + str(self.sPort))
        print("self.dPort : " + str(self.dPort))
        print("self.anomalies : " + str(self.anomalies))

    
    def makeDictWithGraphLabelsAndValues(self, row):
        self.srcIp = row[0]
        self.protocol.append(row[1])
        self.dstIP.append(row[2])
        self.sPort.append(row[3])
        self.dPort.append(row[4])
        self.anomalies = row[5]
        graphletsDict[self.srcIp] = self


    def addFutherNodes(self, row):
        srcIp = row[0]
        graphlet = graphletsDict[srcIp]
        graphlet.protocol.append(row[1])
        graphlet.dstIP.append(row[2])
        graphlet.sPort.append(row[3])
        graphlet.dPort.append(row[4])
        self.printAllAttributs()

    def add_nodes(self, graphlet):
        graphlet.add_node(self.srcIp, name='srcIp')
        for p in self.protocol:
            graphlet.add_node(p, name='protocol')
        for p in self.dstIP:
            graphlet.add_node(p, name='dstIP')
        for p in self.sPort:
            graphlet.add_node(p, name='sPort')
        for p in self.dPort:
            graphlet.add_node(p, name='dPort')
        graphlet.add_node(self.anomalies, name='anomalies')
        print(graphlet.nodes)
        return graphlet
    
    def add_edges(self, graphlet):
        for i in range(1,6):#nb of network flows
            #graphlet.add_nodes_from([933,79,6,21,80])
            node = row[i]
            previousNode = row[i-1]
            graphlet.add_edge(previousNode,node)
            print(graphlet.edges)
            return graphlet
        return graphlet
    
    def constructGraph(self):
        graphlet = nx.DiGraph()
        graphlet = self.add_nodes(graphlet)
        self.calculeRandomWalkKernel(graphlet)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # TO DO -----> modify function above (add_edges) for the moment it does nothing
        #graphlet = self.add_edges(graphlet)

        
################### PART RANDOM KERNERL ###################

        
    def calculeRandomWalkKernel(self, graphlet):
        self.walks0(graphlet)
        self.walks1(graphlet)
        self.walks2()
        self.walks3()
        self.walks4()

    
    def walks0(self, graph):
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
        self.A0 = np.array(matrix)
        print("walk0 : ")
        print(self.A0)



    #adjacency matrix with walks of lengths 1
    def walks1(self, graph):
        matrix = []
        edges = graph.edges
        nodes = graph.nodes
        nbNodes = len(nodes)
        nodesList = list(graph.nodes(data='name'))
        nodesFinales = self.orderingNodes(nodesList)
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
        self.A1 = np.array(matrix)
        print("walk1 : ")
        print(self.A1)

    def multiplicationMatrix(self,A,B):
        return np.matmul(A, B)

    def walks2(self):
        res = self.multiplicationMatrix(self.A1,self.A1)
        self.A2 = np.array(res)
        print("walk2 : ")
        print(self.A2)
    
    
    def walks3(self):
        res = self.multiplicationMatrix(self.A2,self.A1)
        self.A3 = np.array(res)
        print("walk3 : ")
        print(self.A3)


    def walks4(self):
        res = self.multiplicationMatrix(self.A3,self.A1)
        self.A4 = np.array(res)
        print("walk4 : ")
        print(self.A4)

    def sommeWalks(self):
        res = self.A0 + self.A1 + self.A2 + self.A3 + self.A4
        print("somme : ")
        print(res)
        self.somme = res

    def orderingNodes(self, nodesList):
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


######################### READ FILE #########################

def readAnnotatedTrace():
    with open('annotated-trace.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < 29:
                #if line_count == 23 or line_count == 24:
                #graphlet = constructGraph(row)
                #print(graphlet)
                constructObjectGraph(row)

                #                    A0 = calculeMatrixA0(graphlet)
                #                    A = countPathsOfLength1(graphlet)
                #                    A2 = walks2(A)
                #                    A3 = walks3(A2, A)
                #                    A4 = walks4(A3, A)
                #                    sommeWalks(A0,A,A2,A3,A4)
                line_count += 1
            else:
                line_count += 1



######################### CONSTRUCT GRAPHLET OBJECTS #########################

def constructObjectGraph(row):
    srcIp = row[0]
    #if the graphlet with ip in question exist already ->
    if srcIp in graphletsDict:
        #print("===== " + srcIp)
        #print(graphletsDict)
        graphlet = graphletsDict[srcIp]
        #print(graphlet)
        graphlet.addFutherNodes(row)
#addFutherNodes(row)
        #return graphlet
    else:
        g = Graphlet()
        g.makeDictWithGraphLabelsAndValues(row)

######################### CONSTRUCT GRAPHLETS #########################

def constructGraphlets():
    for srcIp,graph in graphletsDict.items():
        graph.constructGraph()



readAnnotatedTrace()
constructGraphlets()
