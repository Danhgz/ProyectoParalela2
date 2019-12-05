import random
import sys, getopt
import math
import time
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

def calcDisPosMin(puntoEnX, centroides, data):
    min = 0.0
    aux = 0.0
    pos = centroides[0]
    min = distance.sqeuclidean(puntoEnX, data[centroides[0]])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, data[centroides[i]])
            if aux < min:
                min = aux
                pos = centroides[i]
    return min, pos
    
def calcDisPosMin2(puntoEnX, centroides, data):
    min = 0.0
    aux = 0.0
    pos = 0
    min = distance.sqeuclidean(puntoEnX, centroides[0])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, centroides[i])
            if aux < min:
                min = aux
                pos = i
    return min, pos

def recluster(data, centroides, vecsXproceso, vCentroides, m, k):
    vContador = [0] * m
    for i in range(vecsXproceso):
        vContador[vCentroides[i]]+= 1
    for i in range(m):
        vContador[i] = comm.allreduce(vContador[i], op= MPI.SUM)
    indicesFinales = np.argpartition(vContador, -k)[-k:]
    centroidesFinales = []
    for i in range(k):
        centroidesFinales.append(data[indicesFinales[i]])
    return centroidesFinales

def init(data,k,m,n):
    centroides = []
    if pid == 0: 
        centroides.append(random.randint(0, m-1))
    centroides = comm.bcast(centroides, 0)
    l = k // 3
    vecsXproceso = m // size
    inicio = pid * vecsXproceso
    fin = inicio + vecsXproceso
    vCostos = np.zeros(vecsXproceso)
    vCentroides = np.zeros(vecsXproceso, dtype= int)
    costo = 0
    for i in range(inicio,fin):
        vCostos[i-inicio], vCentroides[i-inicio] = calcDisPosMin(data[i], centroides, data)
    costo = sum(vCostos)
    costo = comm.allreduce(costo, op=MPI.SUM)
    for i in range(5):
        for j in range(inicio,fin):           
            escoger = random.random()
            prob = (vCostos[j-inicio]*l)/costo
            if escoger < prob: 
                centroides.append(j)
        centroides = comm.gather(centroides,0)
        if pid == 0:
            centroides = [ind for sublist in centroides for ind in sublist]
            centroides = np.unique(centroides)
            centroides = list(centroides)
        centroides = comm.bcast(centroides, 0)   
        for j in range(inicio,fin):
            vCostos[j-inicio], vCentroides[j-inicio] = calcDisPosMin(data[j], centroides, data)
        costo = sum(vCostos)
        costo = comm.allreduce(costo, op=MPI.SUM)
        
    centroidesFinales = recluster(data, centroides, vecsXproceso, vCentroides, m, k)
    return centroidesFinales, costo

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
    posMin = 0
    fCosto = 0.0
    centroidesAlterados = 0
    vecsXproceso = m//size
    inicio = (pid*vecsXproceso)
    fin = inicio + vecsXproceso
    gruposXproceso = k//size
    inicioGpos = (pid*gruposXproceso)
    finGpos = inicioGpos + gruposXproceso
    for i in range(inicio,fin):
        aux, posMin = calcDisPosMin2(vectorDatos[i], vectorCentroides, vectorDatos)
        fCosto += aux
        puntosAsociadosC[posMin].append(vectorDatos[i])
    fCosto = comm.allreduce(fCosto, op=MPI.SUM)
    for i in range(k):
        puntosAsociadosC[i] = comm.allreduce(puntosAsociadosC[i])
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
            aux, posMin = calcDisPosMin2(vectorDatos[i], vectorCentroides, vectorDatos)
            fCostoPrime += aux
            puntosAsociadosC[posMin].append(vectorDatos[i])
        fCostoPrime = comm.allreduce(fCostoPrime, op=MPI.SUM)
        for i in range(k):
            puntosAsociadosC[i] = comm.allreduce(puntosAsociadosC[i])
        if (fCosto - fCostoPrime <= eps) or (centroidesAlterados == 0): #El equivalente a un do-while
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
    print("Tiempo pared: %f   Funcion de Costo: %f"% (tPared,fCosto))
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
    centroides, costo = init(my_data,k,m,n)
    vGrupos, vCentroides, costo = algoritmoLloyd(my_data,centroides, m, n, k, e)
    end = time.time()
    tPared = comm.reduce((end-start), op= MPI.MAX)
    if pid == 0:
        escribirResultados(vGrupos, vCentroides, n, tPared ,costo)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
