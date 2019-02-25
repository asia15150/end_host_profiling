#import sys
import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np

#time, srcIp, protocol, dstIp, sPort, dPort, toBytes


list_graphlets = []
#list_networkFlows = []

class Graphlet:
    def __init__(self):
        #set is to prevent redundancy
        self.ip_adresses = set()
        self.tab_nodes = set()
        self.graphH = nx.MultiDiGraph()
        self.graphG = nx.DiGraph()

    
    def make_graphH(self, srcIp,dstIp, w, c):
        #nodes
        self.graphH.add_node(srcIp)
        self.graphH.add_node(dstIp)
        self.graphH.add_edge(srcIp,dstIp , w, color=c)

    def make_graphG(self):
        #nodes
        self.graphG.add_nodes_from(self.graphH.nodes)
        
        for srcIp, adj in self.graphH.adjacency():
            #print(srcIp)
            #print(adj)
            final_color = ""
            #attr - the dictionary of form {weight:{color:value}}
            for ip, attr in adj.items():
                #print(attr)
                #if there is more than one edge
                if len(attr) > 1:
                    sommeBlue = 0
                    sommeRed = 0
                    for weight, colorList in attr.items():
                        if (colorList["color"] == 'blue'):
                            sommeBlue += int(weight)
                        else:
                            sommeRed += int(weight)
                                #print("sommeBlue ", sommeBlue)
                                #print("sommeRed ", sommeRed)
                    if (sommeBlue > sommeRed):
                        final_color = "blue"
                    else:
                        final_color = "red"
                    self.graphG.add_edge(srcIp,ip, color=final_color)
                else :
                    for weight, colorList in attr.items():
                        final_color = colorList["color"]
                    self.graphG.add_edge(srcIp,ip, color=final_color)
        print(self.graphG.edges.data())




def contruction_part_of_graphlet(g, row):
    ip = row[0]
    srcIp = row[1]
    dstIp = row[3]
    w = row[6]
    sport = int(row[4])
    dport = int(row[5])
    color= "red"
    if (sport <1024 or dport < 1024) :
        color = "blue"
#g.ip_adresses.add(ip)
    g.make_graphH(srcIp, dstIp,w, color)



#read file and build graphlets H and then G
def readAnnotatedTrace(threshold):
    time_threshold = threshold
    with open('graph_printing.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        g = Graphlet()
        for row in csv_reader:
            time = int(row[0])
            if(time < time_threshold ):
                contruction_part_of_graphlet(g, row)
            else :
                list_graphlets.append(g)
                #print(g.ip_adresses)
                g = Graphlet()
                contruction_part_of_graphlet(g, row)
                time_threshold += threshold



#def count_4_star():


def count_2_path(g):


#def count_3_path():



readAnnotatedTrace(30)
#print(list_graphlets)
for g in list_graphlets:
    print(g.graphH.edges)
    print("graphG")
    g.make_graphG()
    print("**********")


