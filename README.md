The goal of the project is to develop a method to carry out anomaly detection in IP traffic.



- building a profile of each IP adress in form of graphlets



CODE EXPLENATIONS:

1) Calling fonction readAnnotatedTrace() which creats the objects "Graphlet" based on data from file "annotated-trace.csv" and initlailise the following attributs :
self.srcIp = -1
self.protocol = []
self.dstIP = []
self.sPort = []
self.dPort = []
self.anomalies = -1

At the same those objects are added to the dictionary "graphletsDict" in form of {srcIp: object Graphlet}

2) We build reall graphs in each object Graphlet

3) We apply algotihm for random walk kernel that return the matrix which is the som of the random walk matrix of length 0,1,2,3 et 4










