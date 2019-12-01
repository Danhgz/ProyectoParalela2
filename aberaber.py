# Proyecto k-means||
#LIBRERIAS
import pprint
import random
import time
import sys, getopt
import math
from mpi4py import MPI
import numpy as np
from numpy import genfromtxt
from scipy.spatial import distance

#VARIABLES GLOBALES
comm = MPI.COMM_WORLD
pid = comm.rank
size = comm.size
#*-*-*-*-*-*-*-*-*-*-*

def leer_consola(argv):
    arch = ""
    k = ""
    m = ""
    n = ""
    e = ""
    try:
        opts, args = getopt.getopt(argv,"ha:k:m:n:e:")
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

def disMin(pos, centroides, data):
    min = 0
    aux = 0
    min = distance.sqeuclidean(data[pos], data[centroides[0]])
    if len(centroides) > 1:
        for i in range(1, len(centroides)):
            aux = distance.sqeuclidean(data[pos], data[centroides[i]])
            if aux < min:
                min = aux
    return min


def recluster(my_data, centroides, n):
    centroidesP = []
    centroidesP.append(centroides[0])
    costo = 0.0
    contador = 1
    for i in range(len(centroides)):
        costo += math.sqrt(disMin(centroides[i], centroidesP, my_data))
    while contador < n:
        posR = random.randint(0, len(centroides)-1)
        prob = random.random()
        escoger = disMin(posR, centroides, my_data)/costo
        if prob < escoger and centroidesP.count(posR) == 0:
            centroidesP.append(posR)
            contador += 1
            costo = 0
            for i in range(len(centroidesP)):
                costo += math.sqrt(disMin(centroides[i], centroidesP, my_data))
    return centroidesP


def init(data, k,m,n):
    centroides = []
    costo = 0
    tam = m//size
    ini = pid*tam
    fin = ini+tam
    if pid == 0:
        centroides.append(0)
    centroides = comm.bcast(centroides, 0)
    for i in range(ini,fin):
        costo += math.sqrt(disMin(i,centroides,data))
    costo = comm.allreduce(costo, op = MPI.SUM)
    l1 = m // 2
    l = math.sqrt(tam)

    for i in range(2):
        count = 0
        while count < l:
            elegir = random.randint(0,m-1)
            prob = (l1 * disMin(elegir, centroides, data))/costo
            print(prob)
            azar = random.random()
            if azar < prob:
                centroides.append(elegir)
                count += 1
                centroides = comm.gather(centroides, 0)
                if pid == 0:
                    centroidesAux = np.array(centroides)
                    centroidesAux = centroidesAux.flatten()
                    #centroidesAux = np.unique(centroidesAux)
                    centroides = list(centroidesAux)
                    print(len(centroides))
                centroides = comm.bcast(centroides,0)
                costo = 0
                for i in range(ini,fin):
                    costo += math.sqrt(disMin(i,centroides,data))
                costo = comm.allreduce(costo, op = MPI.SUM)
    if pid == 0:
        print(centroides)
        centroides = recluster(data, centroides, n)
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
    centroides, costo = init(my_data,k,m,n)
    if pid == 0:
        print(centroides)
        print(costo)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
