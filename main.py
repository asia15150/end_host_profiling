#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:05:42 2019

@author: asia
"""

import networkx as nx
import csv



#examples - exo 3.
#srcIp = ['12.124.65.34' for i in range(0, 4)]
#protocol = [17 for i in range(0, 4)]
#dstIP = ['12.124.65.88' for i in range(0, 4)]
#sPort = [138 for i in range(0, 4)]
#dPort = [2 for i in range(0, 4)] 
#formats = [srcIp, protocol,dstIP, sPort, dPort]

graphlets = {} #dictionary of form ipadress:graphlet


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
#def saveRowInArrays(row):
#    #print(row)
#    srcIp.append(row[0])
#    protocol.append(row[1])
#    dstIP.append(row[2])
#    sPort.append(row[3])
#    dPort.append(row[4])
#    anomalies.append(row[5])    

def addFutherNodes(row):
    print("addFutherNodes : ")
    srcIp = row[0]
    nodes = list (graphlets[srcIp].nodes)
    print(nodes)
    print("nodes[1] " + nodes[1])
    print("row[1] " + row[1])
    if nodes[1] == row[1]:
        print(" 1 : ")
    if nodes[2] == row[2]:
        print(" 2 : ")
    if nodes[3] == row[3]:
        print(" 2 : ")
    if nodes[4] == row[4]:
        print(" 2 : ")


        
   # if row[1] 


    
def constructGraph(row):
    srcIp = row[0]
    graphlet = nx.Graph()
    if srcIp in graphlets:
        print("===== " + srcIp)
        addFutherNodes(row)
    else:
        for i in range(0,5):#nb of network flows
            node = row[i]
            graphlet.add_node(node)
            if i != 0:
                previousNode = row[i-1]
                print(previousNode)
                print(node)
                graphlet.add_edge(previousNode,node) 
                print(graphlet.edges)
#        if i == 0 or i == 1 or i == 2:
#            nx.draw(graphlet)
        #print(graphlet.nodes)
        graphlets[srcIp] = graphlet
        #print(graphlets[srcIp].edges)
    


def readAnnotatedTrace():
    with open('annotated-trace.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < 1:#220:
                constructGraph(row)
                line_count += 1
            else:
                line_count += 1
    #print("number of flows : " + str(line_count))
    #formats = [srcIp, protocol,dstIP, sPort, dPort]
    #return (line_count,formats)
    #return (10,formats)

    #print(formats)

#the array containning the corresponding arrays: srcIp, protocol, dstIP, sPort, dPort, anomalies
readAnnotatedTrace()
#print(graphlets)


graphlet1 = nx.Graph()
graphlet1.add_nodes_from([1,2,3,4,5])
graphlet1.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
graphlet1.add_node(2)
print(graphlet1.edges)
print(graphlet1.nodes)

