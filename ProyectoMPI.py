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

def calcDisPosMin(puntoEnX, centroides):
    min = 0.0
    aux = 0.0
    min = distance.sqeuclidean(puntoEnX, centroides[0])
    pos = 0
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX,centroides[i])
            if aux < min:
                min = aux
                pos = i
    return min , pos

def calcularProb(puntoEnX, centroides, psi, data):
    prob = 0.0
    min = calcDisMin(puntoEnX, centroides, data)
    prob = min / psi
    return prob

def probabilidadesPuntos1(data, centroides, psi, l, m):
    probabilidades = []
    prob = 0.0
    for i in range(m):
        prob = l * calcularProb(data[i], centroides, psi, data)
        probabilidades.append(prob)
    return probabilidades

def probabilidadesPuntos2(data, centroidesF, centroidesIniciales, psi):
    probabilidades = []
    prob = 0.0
    for i in range(len(centroidesIniciales)):
        prob = calcularProb(data[centroidesIniciales[i]], centroidesF, psi, data)
        probabilidades.append(prob)
    return probabilidades

def reclusterKmeans2(my_data, centroides, n):
    centroidesP = []
    centroidesP.append(centroides[0])
    costo = 0.0
    contador = 1
    for i in range(len(centroides)):
        costo += calcDisMin(my_data[centroides[i]], centroidesP, my_data)
    #n = n//size
    while contador < n:
        probabilidades = probabilidadesPuntos2(my_data, centroidesP, centroides, costo)
        posR = random.randint(0, len(centroides)-1)
        prob = random.random()
        if prob < probabilidades[posR] and centroidesP.count(posR) == 0:
            centroidesP.append(posR)
            contador += 1
            costo = 0
            for i in range(len(centroidesP)):
                costo += calcDisMin(my_data[centroides[i]], centroidesP, my_data)
    return centroidesP


def initParalelo(my_data, k, m, n):
    centroides = []
    l = m // 2
    fCosto = 0.0
    centroides.append(random.randint(0,m-1)) 
    l2 = l // size
    probabilidades = []
    if pid == 0:
        for i in range(k):
            fCosto += calcDisMin(my_data[i], centroides, my_data)
        probabilidades = probabilidadesPuntos1(my_data, centroides, fCosto, l, m)
    fCosto = comm.bcast(fCosto, 0)
    probabilidades = comm.bcast(probabilidades, 0)
    contador = 0

    for i in range(5):
        while contador < l2:
            escoger = random.random()
            posicionR = random.randint(0, len(probabilidades)-1)
            if escoger < probabilidades[posicionR]:
                centroides.append(posicionR)
                contador += 1
        fCosto = 0.0
        contador = 0
        for i in range(m):
            fCosto += calcDisMin(my_data[i], centroides, my_data)
        probabilidades = probabilidadesPuntos1(my_data, centroides, fCosto, l, m)

    centroides = comm.gather(centroides, 0)
    centroidesAux = []
    centroidesFinales = []

    if pid == 0:
        for i in range(size):
            for j in range(l2):
                centroidesAux.append(centroides[i][j])
        centroidesAux = list(dict.fromkeys(centroidesAux))
        centroidesFinales = reclusterKmeans2(my_data, centroidesAux, n)
    
    centroidesFinales = comm.bcast(centroidesFinales, 0)
    for i in range(len(centroidesFinales)):
        centroidesFinales[i] = my_data[centroidesFinales[i]]
    return centroidesFinales

def promediarGrupo(vectorGrupo, vectorPromedio, tamDato):
    npGrupo = np.array(vectorGrupo)
    tamGrupo = len(vectorGrupo)
    anterior = vectorPromedio
    cambio = 1
    np.mean(npGrupo, axis = 0, out = vectorPromedio)
    if np.array_equal(anterior, vectorPromedio):
        cambio = 0
    return cambio, vectorPromedio


def algoritmoLloyd(vectorDatos, vectorCentroides, m:int,n:int, k:int, eps:float) :
    puntosAsociadosC = [[] for i in range(k)]
    vectorCentroides.append(vectorDatos[3])
    posMin = 0
    fCosto = 0.0
    centroidesAlterados = 0
    vecsXproceso = m//size
    gruposXproceso = k//size
    inicioGpos = (pid*gruposXproceso)
    finGpos = inicioGpos + gruposXproceso
    inicio = (pid*vecsXproceso)
    fin = inicio + vecsXproceso
    for i in range(inicio,fin):
        aux, posMin = calcDisPosMin(vectorDatos[i], vectorCentroides)
        fCosto += aux
        puntosAsociadosC[posMin].append(vectorDatos[i])
    for i in range(k):
        puntosAsociadosC[i] = comm.allreduce(puntosAsociadosC[i])
    fCosto = comm.allreduce(fCosto, op=MPI.SUM)
    fCostoPrime = fCosto
    while True: #El equivalente a un do-while
        fCosto = fCostoPrime
        centroidesAlterados = 0 
        for i in range(inicioGpos,finGpos):
            centroidesAlterados, vectorCentroides[i] = promediarGrupo(puntosAsociadosC[i], vectorCentroides[i], n)
        centroidesAlterados = comm.allreduce(centroidesAlterados, op=MPI.SUM)
        puntosAsociadosC = [[] for i in range(k)]
        fCostoPrime = 0.0
        for i in range(inicio,fin):
            aux, posMin = calcDisPosMin(vectorDatos[i], vectorCentroides)
            fCostoPrime += aux
            puntosAsociadosC[posMin].append(vectorDatos[i])
        for i in range(k):
            puntosAsociadosC[i] = comm.allreduce(puntosAsociadosC[i])
        fCostoPrime = comm.allreduce(fCostoPrime, op=MPI.SUM)
        if ((fCosto - fCostoPrime) <= eps) or (centroidesAlterados == 0): #El equivalente a un do-while
            break
    return puntosAsociadosC, vectorCentroides, fCostoPrime

def escribirResultados(vGrupos, centroides, n:int, tPared, fCosto) :
    cantGrupos = len(vGrupos)
    salida = open("salida.csv", "w+")
    for i in range(0,cantGrupos) :
        salida.write("Centroide:\n")
        for j in range(0,n) :
            salida.write("%f" % centroides[i][j])
            if (j != n - 1) :
                salida.write(", ")

        salida.write("\nDatos:\n")
        for j in range(0,len(vGrupos[i])) :
            for k in range(0,n) :
                salida.write("%f" % vGrupos[i][j][k])
                if (k != n - 1) :
                    salida.write(", ")
            salida.write("\n")
        salida.write("Elementos: %d\n\n" % len(vGrupos[i]))

    salida.write("Tiempo pared: %f   Funcion de Costo: %f"% (tPared,fCosto))
    salida.close()
    return

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
    start = time.time()
    centroides = initParalelo(my_data, k, m, n)
    medio = time.time()
    #pprint.pprint(centroides)
    medio2 = time.time()
    vGrupos, vCentroides, fCosto = algoritmoLloyd(my_data, centroides, m, n, k, e)
    end = time.time()
    tPared = end-medio2 + medio - start
    tPared = comm.reduce(tPared, op= MPI.MAX)
    if pid == 0 :
        escribirResultados(vGrupos, vCentroides, n, tPared, fCosto)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
