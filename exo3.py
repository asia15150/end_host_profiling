import main as m
import matplotlib.pyplot as plt
import pandas as pd



plt.figure()

graphlets_ = {}

def draw2(fig, g):
    df = pd.DataFrame(g, columns=g.graph.nodes(), index=g.graph.nodes())
    #A = np.squeeze(np.asarray(B))
    plt.table(cellText=df.values,
              rowLabels=df.index, colLabels=df.columns,
              loc='center', fontsize=30)
    plt.axis('off')
    fig.savefig('plots/'+g.ip_adress+'.png', dpi=200)
    plt.close(fig)

row1 = ["12.124.65.34", "17", "12.124.65.33", "138" ,"138"]
row2 = ["12.124.65.35", "17", "12.124.65.37", "80" ,"80"]
row3 =["12.124.65.35" ,"6", "12.124.65.36" ,"167" ,"80"]
row4 =["12.124.65.36", "6", "12.124.65.37" ,"443", "443"]
list_rows = [row1, row2, row3, row4]

#G = nx.path_graph(row1)
#print("GGGGG " + str(G))
# return set(list(G.edges)) #return set just to be unioned later on

for row in list_rows:
    ip = 'srcIp:'+row[0]
    G = graphlets_.get(ip)
    if G == None:
        G = m.Graphlet(ip)
        G.saveRowInArrays(row)
        graphlets_.update({ip:G})
    else:
        G.saveRowInArrays(row)
    G.make_graph()
#print(G.graph.nodes)
#   print(G.graph.edges)
#G.draw()

#G.draw_v2()
    fig = m.draw(G)
#G.draw()
#   draw2( fig, G)
#
plt.show()
print(graphlets_)
for c,g in graphlets_.items():
    print("Graph with id ", c)
    print("Nodes : ",g.graph.nodes)
    print("Edges : ",g.graph.edges,"\n")

