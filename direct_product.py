import final as fn
import csv
import numpy as np



def damping_factor(direct_product_matrix):
    #print(direct_product_matrix)
    maxInDegree = 0
    maxOutDegree = 0
    sum_all_columns = direct_product_matrix.sum(axis=0)
    #print(sum_all_columns[0,0])
    sum_all_row = direct_product_matrix.sum(axis=1)
    #print(sum_all_row)
    l = len(direct_product_matrix)
    for index in range(l):
        maxInDegree = max(maxInDegree, sum_all_columns[index])
        #print(maxInDegree)
        maxOutDegree = max(maxOutDegree, sum_all_row[index])
    #print(maxOutDegree)
    #print(maxInDegree)
    #print(maxOutDegree)
    
    factor = 1/min(maxInDegree,maxOutDegree)
    return factor




def direct_product_kernel(matrix_adj_A, matrix_adj_B):
    l_A = len(matrix_adj_A)#size of matrix adjacency A
    l_B = len(matrix_adj_B)#size of matrix adjacency B
    
    #print(np.array(matrix_adj_A))
    #print(np.array(matrix_adj_B))
    products = None
    
    ######the result of [A, A] direct product [A, A] is
    #[AA, AA, AA, AA]
    for i in range(l_A):#for each column of A
        matrix = None
        for j in range(l_A):#for each row of A
            #multiply an element of A with the whole matrix of B
            new_matrix = matrix_adj_A[i,j] * matrix_adj_B #multiply an element of
            
            #we concatenate the matrix
            if j == 0:# we initialise new line with the first chunk
                matrix = new_matrix
            else:
                matrix = np.concatenate((matrix, new_matrix), axis=1)
    #we concatenate the next chunk to the same line
    if i == 0:#concatenate by column
        products = matrix
        #print(products)
    else:
        products = np.concatenate((products, matrix), axis=0)
#print(matrix)
#print("\n\n\nDIRECT PRODUCT MATRIX \n\n",matrix)


    #we now can compute the goemetric sum
    #(I - dpf*direct_product_matrix)exposant(-1))
    dpf = damping_factor(products)
    #print("\n\nDAMPING FACTOR : ", dpf,"\n\n")

    I = np.identity(len(products))#matrix identity

    k = np.linalg.inv(np.subtract(I, dpf*products))#inversion
    return np.array(k).sum()



#************************* VOL 2 ************************************************** **************************************************
def main_direct_product():
    adjacencies = []
    anomalie = None
    normal = None
    anomalies = []
    for index, g in enumerate(graphlets_.values()):
        g.make_graph()
        g.make_first_matrix()
        adjacencies.append(np.array(g.get_first_matrix()))
        if(g.anomalie != -1):
            if "normal" in g.anomalie:
                normal = np.array(g.get_first_matrix())
            else:
                print(g.ip_adress)
                print(index)
                anomalie = np.array(g.get_first_matrix())
                anomalies.append(anomalie)

#print(adjacencies)
#dpk = direct_product_kernel(adjacencies[0],adjacencies[1])
#print(normal.shape)
#print(anomalie.shape)
#dpk = direct_product_kernel(normal,anomalie)


#list = np.zeros((1,200))
#for i in range(1):
#   for j in range(200):
#        list[i,j] = (int(direct_product_kernel(adjacencies[14],adjacencies[j])))
#print(list)

#direct = direct_product_kernel_v2(adjacencies[14],adjacencies[j])
#print(direct)

#dpk = direct_product_kernel(adjacencies[10],adjacencies[100])
#print(list)
#print("RESULT AFTER SUMMING UP : ",dpk,"\n\n")
#l = len(adjacencies)
#kernel_matrix = np.zeros((l,l))
#print(kernel_matrix.shape)
#time_start = time.process_time()
#run your code


graphlets_ = fn.readTrace('annotated-trace.csv')
graphlets_not = fn.readTrace('not-annotated-trace.csv')


def damping_factor_v2(direct_product_matrix):
    maxInDegree = np.amax(direct_product_matrix.sum(axis=0))#somme columns then find max
    maxOutDegree = np.amax(direct_product_matrix.sum(axis=1))#somme row then find max
    factor = 1/min(maxInDegree,maxOutDegree)
    
    #print('DAMPING-FACTOR--------',factor,'---------------')
    return factor

def direct_product_kernel_v2(matrix_adj_A, matrix_adj_B):
    lA = len(matrix_adj_A)
    lB = len(matrix_adj_B)
    size = lA*lB
    direct_product = np.einsum('ij,kl->ikjl',matrix_adj_A,matrix_adj_B).reshape(size,size)#direct product or tendordot then reshape
    #print(direct_product)
    dpf = damping_factor_v2(direct_product)
    
    I = np.identity(size)#matrix identity
    
    k = np.linalg.inv(np.subtract(I, dpf*direct_product)).sum()#inversion
    #print('Direct Product Sum Kernel Value--------',k,'---------------')
    return k

def main_direct_product_v2():
    adjacencies = []
    anomalie = None
    normal = None
    anomalies = []
    for index, g in enumerate(graphlets_.values()):
        g.make_graph()
        g.make_first_matrix()
        adjacencies.append(np.array(g.get_first_matrix()))
        if(g.anomalie != -1):
            if "normal" in g.anomalie:
                normal = np.array(g.get_first_matrix())
            else:
                #print(g.ip_adress)
                #print(index)
                anomalie = np.array(g.get_first_matrix())
                anomalies.append(anomalie)
list_ = np.zeros((100,1001))



#time_start = time.process_time()
#for i in range(2):
#Parallel(n_jobs=2)(delayed(direct_product_kernel_v2)(adjacencies[i],adjacencies[j]) for j in range(1001))
#time_end = time.process_time()
#print(list_)
#print('time::::',time_end-time_start)




main_direct_product_v2()


#main_direct_product()
