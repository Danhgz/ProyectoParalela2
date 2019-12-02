import pprint
import random
import sys, getopt
import math
from mpi4py import MPI
import numpy as np
from numpy import genfromtxt
from scipy.spatial import distance

comm = MPI.COMM_WORLD
size = comm.size
pid = comm.rank


def leer_consola(argv):
    arch = ""
    k = ""
    m = ""
    n = ""
    e = ""
    try:
        opts, args = getopt.getopt(argv,"h:a:k:m:n:e:")
    except getopt.GetoptError:
        print ('Kmeans||.py -a <archivo> -k <valor de k> -m <valor de C> -n <valor de N> -e <Valor de epsilon>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Kmeans||.py -a <archivo> -k <valor de k> -m <valor de C> -n <valor de N> -e <Valor de epsilon>')
            sys.exit()
        elif opt == "-a":
            arch = arg
        elif opt == "-k":
            k = arg
        elif opt == "-m":
            m = arg
        elif opt == "-n":
            n = arg
        elif opt == "-e":
            e = arg
    return str(arch), int(k), int(m), int(n), int(e)


def calcDisMin(puntoEnX, centroides, data):
    min = 0.0
    aux = 0.0
    min = distance.sqeuclidean(puntoEnX, data[centroides[0]])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, data[centroides[i]])
            if aux < min:
                min = aux
    return min

def calcPosMin(puntoEnX, centroides, data):
    min = 0.0
    aux = 0.0
    pos = 0
    min = distance.sqeuclidean(puntoEnX, data[centroides[0]])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, data[centroides[i]])
            if aux < min:
                min = aux
                pos = i
    return pos
def calcularProb(puntoEnX, centroides, psi, data):
    prob = 0.0
    min = calcDisMin(puntoEnX, centroides, data)
    prob = (math.pow(min,2)) / psi
    return prob

def recluster(my_data, centroides, n, costo):
    centroidesP = []
    centroidesP.append(random.randint(0,len(centroides)-1))
    contador = 1

    while contador < n:
        posR = random.randint(0, len(centroides)-1)
        prob = random.random()
        escoger = math.pow(calcDisMin(posR, centroides, my_data),2)/costo
        if prob < escoger and centroidesP.count(posR) == 0:
            centroidesP.append(posR)
            contador += 1
            costo = 0
            for i in range(len(centroides)):
                costo += calcDisMin(centroides[i], centroidesP, my_data)
    return centroidesP

def init(data,k,m,n):
    centroides = []
    l = n // 2
    tam = m // size
    costo = 0
    centroides.append(0)
    recv = np.zeros((tam, k))
    comm.Scatter(data, recv, root = 0)
    ini = pid * tam
    fin = ini+tam

    for i in range(tam):
           costo += calcDisMin(recv[i], centroides, data)
    costo = comm.allreduce(costo, op=MPI.SUM)

    for i in range(5):
        cont = 0
        while cont < l:
            rand = random.randint(0,tam-1)
            escoger = random.random()
            prob = l * calcularProb(recv[rand], centroides,costo,data)
            if escoger < prob:
                print("escoger: " +str(escoger))
                print("prob: " +str(prob))
                cont += 1
                pos = rand + ini
                centroides.append(pos)
                centroides = comm.allgather(centroides)
                centroides = np.array(centroides)
                centroides = centroides.flatten()
                centroides = np.unique(centroides)
                centroides = list(centroides)
                costo = 0
                for j in range(tam):
                       costo += calcDisMin(recv[j], centroides, data)
                costo = comm.allreduce(costo, op=MPI.SUM)
    # if pid == 0:
    #     centroides = recluster(data, centroides,n,costo)
    # centroides = comm.bcast(centroides, 0)
    # pesos = [0] * len(centroides)
    # pos = 0
    # for i in range(tam):
    #     pos = calcPosMin(recv[i], centroides, data)
    #     pesos[pos] += 1
    # for i in range(len(pesos)):
    #     pesos[i] = comm.allreduce(pesos[i], op=MPI.SUM)
    # print(pesos)
    # centroidesFinales = []
    # maxi = 0
    # pos = 0
    # if pid == 0:
    #     c = 0
    #     while c < n:
    #         maxi = max(pesos)
    #         pos = pesos.index(maxi)
    #         if centroidesFinales.count(pos) == 0:
    #             centroidesFinales.append(pos)
    #             c += 1
    #         pesos.remove(maxi)
    # centroidesFinales = comm.bcast(centroidesFinales, 0)
    for i in range(len(centroides)):
        centroides[i] = data[centroides[i]]
    return centroides, costo

def main (argv):
    arch = ""
    k = 0
    m = 0
    n = 0
    e = 0
    if pid == 0:
        arch, k, m, n, e = leer_consola(argv)
    arch, k, m, n, e = comm.bcast((arch,k, m, n, e), root = 0)
    my_data = genfromtxt(arch, delimiter=',')
    centroides,costo = init(my_data,k,m,n)
    if pid == 0:
        print(len(centroides))

    return

if __name__ == "__main__":
    main(sys.argv[1:])

# def main():
# 	comm = MPI.COMM_WORLD
# 	size = comm.size
# 	pid = comm.rank
#
# 	k = 20
# 	recvBuf = np.zeros(k//size)
# 	sendBuf = np.zeros(k//size)
#
# 	if pid == 0:
# 		sendBuf = np.random.rand(k) # crea un arreglo aleatorio
#
# 	# cada uno de los procesos recibe k//size elementos
# 	comm.Scatter([sendBuf, MPI.DOUBLE],[recvBuf, MPI.DOUBLE],0)
# 	pprint.pprint(("proceso %i recibe %s")%(pid,np.array2string(recvBuf)))
#
# 	# todos multiplican por un escalar el arreglo :
# 	recvBuf = recvBuf*2
# 	comm.Gather([recvBuf, MPI.DOUBLE],[sendBuf, MPI.DOUBLE],0)
# 	if pid == 0:
# 		pprint.pprint(("proceso %i recibe %s")%(0,np.array2string(sendBuf)))
# main()
