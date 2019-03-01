#import sys
import pandas as pd
import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

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
        self.graphlet_count = {}
        self.orbit_count = {}

    
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

    
    def drawG(self):
        plt.subplot(111)
        G = self.graphG
        pos = nx.circular_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        nx.draw(G, with_labels=True, edges=edges, edge_color=colors)
        plt.show()
    
    def drawH(self):
        G = self.graphH
        plt.subplot(111)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()


    
    def count_2_path(self):
        return len(list(self.graphG.edges))

    def count_3_path(self):
        res = 0
        G = self.graphG
        #nodes = list(g.graphG.nodes)
        for node in G.nodes:
            #print(node, " : ")#, " : ", list(G.adj[node]))
            for adj in G.adj[node]:
                #print (adj)
                for adj_2 in G.adj[adj]:
                    #print ("     -     ", adj_2)
                    res +=1
        return res

    def count_4_star(self):
        res = 0
        G = self.graphG
        for node in G.nodes:
            for adj in G.adj[node]:
                size_addjacance_list = len(list(G.adj[adj]))
                if ( size_addjacance_list == 3):
                    res += 1
                elif size_addjacance_list > 3:
                    res += (size_addjacance_list+ 1)
        return res
    
    
    def count_3_path_interior(self, node):
        G = self.graphG
        adjacency_size = len(list(G.adj[node]))
        nb_parent_nodes = 0
        if adjacency_size > 0:
            for vert in G:
                adj_list = G.adj[vert]
                nb_parent_nodes += len(adj_list)
        return nb_parent_nodes * adjacency_size

    
    def count_3_path_terminal(self, node):
        G = self.graphG
        res = 0
        for adj in G.adj[node]:
            #print (adj)
            for adj_2 in G.adj[adj]:
                #print ("     -     ", adj_2)
                res +=1
                    
        for vert in G:
            adj_list = G.adj[vert]
            for v in G:
                for vv in adj_list:
                    if (vv in G.adj[v]):
                        res+=1
        return res
            
    
    def count_4_star_center(self, node):
        G = self.graphG
        size_adj = len(list(G.adj[node]))
        if size_adj == 4:
            return 1
        if size_adj >=4:
            return size_adj
        return 0

    def count_4_star_leaf(self, node):
        G = self.graphG
        list_parent_nodes = G.adj[node]
        res = 0
        for nd in list_parent_nodes:
            if len(list(G.adj[nd])) >= 4:
                res +=1
        return res
    

    def compute_graphlet_count(self):
        #print(self.graphH.edges)
        self.make_graphG()
        self.graphlet_count["2_path"] = self.count_2_path()
        self.graphlet_count["3_path"] = self.count_3_path()
        self.graphlet_count["4_star"] = self.count_4_star()


    def compute_orbit_count(self):
        G = g.graphG
        for node in G.nodes:
            orbit_c = {}
            orbit_c["3_path_interior"] = self.count_3_path_interior(node)
            orbit_c["3_path_terminal"] = self.count_3_path_terminal(node)
            orbit_c["4_star_center"] = self.count_4_star_center(node)
            orbit_c["4_star_leaf"] = self.count_4_star_leaf(node)
            self.orbit_count[node] = orbit_c
            return 0




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
#where the threshold is the he number of seconds between 2 diffrent time windows
# for instance: threshold = 30 signifies: time window 1: 0-30, time window 2 : 30-60
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





readAnnotatedTrace(30)
#print(list_graphlets)
for g in list_graphlets:
    g.compute_graphlet_count()
    g.compute_orbit_count()
    g.drawH()
    g.drawG()
    
    print ("graphlet count ", name_node,)
    for name_struct, nb in g.graphlet_count.items():
        print(" ",name_struct, " : ", nb)
    
    for name_node, dict in g.orbit_count.items():
        print ("orbit count for node ", name_node,)
        for struct, nb_ in dict.items():
            print(" ",struct, " : ", nb_)
    print("******************************************************************************************")



