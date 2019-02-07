#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:05:42 2019

@author: asia
"""

import networkx as nx
import csv
import matplotlib.pyplot as plt
from random import choice


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
tab_nodes = []



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



def makeNodes_v2(tab):
    array = []
    #print(tab)
    G = nx.path_graph(tab)
    #H = G.subgraph(tab)
    #print(G.edges)
    #l = len(tab)
    #print(l)
    #for i in range(0, l-1):
     #   array.append((tab[i],tab[i+1]))
    #array.append((tab[0],tab[l-1]))
    #print(list(H.edges))
    return list(G.edges)


def saveRowInArrays(row):
    #print(row)
    row[0] = 'srcIp:'+row[0]
    row[1] = 'protocol:'+row[1]
    row[2] = 'dstIP:'+row[2]
    row[3] = 'sPort:'+row[3]
    row[4] = 'dPort:'+row[4]
    row[5] = 'anomalies:'+row[5]
    srcIp.append(row[0])
    protocol.append(row[1])
    dstIP.append(row[2])
    sPort.append(row[3])
    dPort.append(row[4])
    anomalies.append(row[5])
    tab = makeNodes_v2(row)
    global tab_nodes
    tab_nodes = tab_nodes + tab

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
    for i in range(0, l-3):
        array.append((tab[i],tab[i+1]))
    #print(array)
    return array


        
def readAnnotatedTrace():
    tab = []
    with open('annotated-trace.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            saveRowInArrays(row)
            #row.pop()
            #tab = tab + row;
            if line_count < 100:
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
def draw(tab):
    for t in tab:
        graphlet1 = nx.Graph()
        graphlet1.add_nodes_from(tab)
        pos = nx.spring_layout(graphlet1)
        nx.draw_networkx_nodes(graphlet1, pos,nodelist=t,node_color='r',node_size=100)
    plt.axis('off')
    plt.show()
    
#the array containning the corresponding arrays: srcIp, protocol, dstIP, sPort, dPort, anomalies

#print(graphlets)




print(tab_nodes)

#graphlet1.add_nodes_from(srcIp, color="red")
#graphlet1.add_nodes_from(protocol, color="blue")
#graphlet1.add_nodes_from(dstIP, color='yellow')
#graphlet1.add_nodes_from(sPort, color='black')
#graphlet1.add_nodes_from(dPort, color='green')
#graphlet1.add_nodes_from(anomalies, color='pink')




#graphlet1.add_edges_from(finalTab)

def drawG():
    tab = readAnnotatedTrace()
    finalTab = makeNodes(tab)
    graphlet1 = nx.Graph()
    graphlet1.add_nodes_from(srcIp, color="red")
    graphlet1.add_nodes_from(protocol, color="blue")
    graphlet1.add_nodes_from(dstIP, color='yellow')
    graphlet1.add_nodes_from(sPort, color='black')
    graphlet1.add_nodes_from(dPort, color='green')
    graphlet1.add_nodes_from(anomalies, color='pink')
    pos = nx.spring_layout(graphlet1)
    graphlet1.add_edges_from(tab_nodes)

    red_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='red']
    blue_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='blue']
    yellow_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='yellow']
    black_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='black']
    green_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='green']
    pink_nodes=[n for n,d in graphlet1.nodes(data=True) if d['color']=='pink']

    nx.draw_networkx_nodes(graphlet1,pos,nodelist=blue_nodes,node_color='b', node_size=500)
    nx.draw_networkx_nodes(graphlet1,pos,nodelist=red_nodes,node_color='r', node_size=500)
    nx.draw_networkx_nodes(graphlet1,pos,nodelist=yellow_nodes,node_color='green', node_size=500)
    nx.draw_networkx_nodes(graphlet1,pos,nodelist=black_nodes,node_color='deepskyblue', node_size=500)
    nx.draw_networkx_nodes(graphlet1,pos,nodelist=green_nodes,node_color='yellow', node_size=500)
    nx.draw_networkx_nodes(graphlet1,pos,nodelist=pink_nodes,node_color='pink', node_size=500)
    nx.draw_networkx_edges(graphlet1,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(graphlet1,pos,font_color='black')

    plt.axis('off')
    plt.show()

def random_walk(G):
    #G.nodes(data=False)
    #print(choice(G.nodes()))
    #print(G)
    i = 0
    contador = 0
    execucoes = 0
    
        
def drawG2():
    tab = readAnnotatedTrace()
    finalTab = makeNodes(tab)

    graphlet1 = nx.Graph()
    
    graphlet1.add_nodes_from(srcIp, label = 'srcIp')
    graphlet1.add_nodes_from(protocol, label = 'protocol')
    graphlet1.add_nodes_from(dstIP, label = 'dstIP')
    graphlet1.add_nodes_from(sPort, label = 'sPort')
    graphlet1.add_nodes_from(srcIp, label = 'srcIp')
    graphlet1.add_nodes_from(dPort, label = 'dPort')
    graphlet1.add_nodes_from(anomalies, label = 'anomalies')
    graphlet1.add_edges_from(tab_nodes)
    graphlet1.nodes(data=True)
    color_map = []
    for node in graphlet1:
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
        
    random_walk(graphlet1)
    nx.draw(graphlet1, with_labels=True,node_color = color_map, node_size=500)
    plt.axis('off')
    plt.show()




#print(srcIp)
#print(protocol)
#print(dstIP)
#print(sPort)
#print(dPort)
#print(anomalies)
conc = [srcIp,protocol,dstIP, sPort, dPort]

drawG2()

#graphlet1.add_nodes_from(tab)
#graphlet1.add_edges_from(tab_nodes)

#print(tab_nodes)
#graphlet1.add_nodes_from(srcIp, color="red")
#graphlet1.add_nodes_from(protocol, color="blue")
#graphlet1.add_nodes_from(dstIP, color='yellow')
#graphlet1.add_nodes_from(sPort, color='black')
#graphlet1.add_nodes_from(dPort, color='green')

#print(graphlet1.nodes(data=True))

#nx.draw(graphlet1)
#nx.draw_networkx_nodes(graphlet1, pos)
#plt.axis('off')
#plt.show()
#print(graphlet1.nodes)

