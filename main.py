#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:05:42 2019

@author: asia
"""

import networkx as nx
import csv
import matplotlib.pyplot as plt


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
def saveRowInArrays(row):
    #print(row)
    srcIp.append(row[0])
    protocol.append(row[1])
    dstIP.append(row[2])
    sPort.append(row[3])
    dPort.append(row[4])
    anomalies.append(row[5])    

def addFutherNodes(row):
    print("addFutherNodes : ")
    srcIp = row[0]
    graphlet = graphlets[srcIp]
    nodes = list (graphlet.nodes)
    print(nodes)
    print("nodes[1] " + nodes[1])
    print("row[1] " + row[1])
    for i in range(1,5):#nb of network flows
            node = row[i]
            graphlet.add_node(node)
            #print(graphlet.nodes)
            previousNode = row[i-1]
                #print(previousNode)
                #print(node)
            graphlet.add_edge(previousNode,node)
    #print(graphlet.nodes)
    print("?")
    print(graphlet.edges)


#    if nodes[1] == row[1]:
#        print(" 1 : ")
#    if nodes[2] == row[2]:
#        print(" 2 : ")
#    if nodes[3] == row[3]:
#        print(" 2 : ")
#    if nodes[4] == row[4]:
#        print(" 2 : ")


        
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
                #print(previousNode)
                #print(node)
                graphlet.add_edge(previousNode,node) 
                #print(graphlet.edges)
                #print(graphlet.nodes)

#        if i == 0 or i == 1 or i == 2:
#            nx.draw(graphlet)
        #print(graphlet.nodes)
        graphlets[srcIp] = graphlet
        #print(graphlets[srcIp].edges)
    
def makeNodes(tab):
    array = []
    l = len(tab)
    #print(l)
    for i in range(0, l-1):
        array.append((tab[i],tab[i+1]))

    array.append((tab[0],tab[l-1]))
    #print(array)
    return array
        
def readAnnotatedTrace():
    tab = []
    with open('annotated-trace.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            saveRowInArrays(row)
            if line_count < 10:
                tab = tab + row;
                #print(tab)
                line_count+=1
            else:
                break
    return tab
            #if line_count < 220:
             #   constructGraph(row)
              #@  line_count += 1
            #else:
             #   line_count += 1
    #print("number of flows : " + str(line_count))
    #formats = [srcIp, protocol,dstIP, sPort, dPort]
    #return (line_count,formats)
    #return (10,formats)

    #print(formats)

    
#the array containning the corresponding arrays: srcIp, protocol, dstIP, sPort, dPort, anomalies

#print(graphlets)

tab = readAnnotatedTrace()
finalTab = makeNodes(tab)

graphlet1 = nx.Graph()
graphlet1.add_nodes_from(srcIp, color="red")
graphlet1.add_nodes_from(protocol, color="blue")
graphlet1.add_nodes_from(dstIP, color='yellow')
graphlet1.add_nodes_from(sPort, color='black')
graphlet1.add_nodes_from(dPort, color='green')
graphlet1.add_edges_from(finalTab)

print(srcIp)
print(protocol)
print(dstIP)
print(sPort)
print(dPort)
print(anomalies)



#nx.draw_networkx_edges(graphlet1,pos,width=1.0,alpha=0.5)
nx.draw(graphlet1)
#nx.draw_networkx_nodes(graphlet1, pos)
plt.show()
#print(graphlet1.nodes)

