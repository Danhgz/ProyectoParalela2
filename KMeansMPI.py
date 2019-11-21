# Proyecto k-means||
#LIBRERIAS
import pprint
import random
import sys, getopt
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
        print ('Kmeans||.py -k <valor de k> -m <valor de C> -n <valor de N> -e <Valor de epsilon>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Kmeans||.py -k <valor de k> -m <valor de C> -n <valor de N> -e <Valor de epsilon>')
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

def calcDisMin(puntoEnX, centroides):
    min = 0.0
    aux = 0.0
    min = distance.sqeuclidean(puntoEnX, centroides[0])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, vectorCentroides[i])
            if aux < min:
                min = aux
    return min

def initParalelo(my_data, k, m, n):
    centroides = []
    if pid == 0:
        posPrimerC = np.random.randint(m) #genera un numero entero aleatorio entre 0 y m exclusivo
    posPrimerC = comm.bcast(posPrimerC, 0)
    centroides.append(posPrimerC)
    fCosto = 0.0
    vecsXproceso = m//size
    inicio = (pid*vecsXproceso)
    fin = inicio + vecsXproceso
    for i in range(inicio,fin):
        fCosto += calcDisMin(my_data[i], centroides)
    fCosto = comm.allreduce(fCosto, op=MPI.SUM)
    return centroides

def main (argv):
    print("aaa")
    if pid == 0:
        arch, k, m, n, e = leer_consola(argv)
    arch, k, m, n, e = comm.bcast((arch,k, m, n, e), 0)
    my_data = genfromtxt(arch, delimiter=',')
    print(my_data," ",m," ",n," ",e)
    centroides = initParalelo(my_data, k, m, n)  
    return

if __name__ == "__main__":
    main(sys.argv[1:])