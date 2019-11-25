# Proyecto k-means||
#LIBRERIAS
import pprint
import random
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

def calcDisMin(puntoEnX, centroides):
    min = 0.0
    aux = 0.0
    min = distance.sqeuclidean(puntoEnX, centroides[0])
    if len(centroides) > 1 :
        tam = len(centroides)
        for i in range(1,tam):
            aux = distance.sqeuclidean(puntoEnX, centroides[i])
            if aux < min:
                min = aux
    return min
def calcularProb(puntoEnX, centroides, psi):
    prob = 0.0
    min = math.sqrt(calcDisMin(puntoEnX, centroides))
    prob = (min / psi)
    return prob

def probabilidadesPuntos(my_data, centroides, psi, l, inicio, fin):
    probabilidades = []
    prob = 0.0
    for i in range(inicio,fin):
        prob = l * calcularProb(my_data[i], centroides, psi)
        probabilidades.append(prob)
    probabilidadesF = np.array(probabilidades)
    return probabilidadesF

def reclusterKmeans2(my_data,centroides,k):

def initParalelo(my_data, k, m, n):
    centroides = []
    l = m // 2
    posPrimerC = 0
    if pid == 0:
        posPrimerC = random.randint(0, m) #genera un numero entero aleatorio entre 0 y m exclusivo
    posPrimerC = comm.bcast(posPrimerC, 0)
    centroides.append(posPrimerC)
    fCosto = 0.0
    vecsXproceso = m//size
    inicio = (pid*vecsXproceso)
    fin = inicio + vecsXproceso
    for i in range(inicio,fin):
        fCosto += calcDisMin(my_data[i], centroides)
    fCosto = comm.allreduce(fCosto, op=MPI.SUM)
    probabilidades = probabilidadesPuntos(my_data, centroides, fCosto, l , inicio, fin)
    l2 = l//size
    contador = 0
    for i in range(0,5):
        probabilidades = probabilidadesPuntos(my_data, centroides, fCosto, l , inicio, fin)
        while contador < l2:
            escoger = random.random()
            posicionR = random.randint(0,probabilidades.size-1)
            if escoger < probabilidades[posicionR] and centroides.count(posicionR) == 0:
                centroides.append(posicionR)
                contador += 1

    centroidesFinales = comm.gather(centroides, 0)
        #luego sigue un allgather
    return centroidesFinales


def promediarGrupo(vector<vector<double>>& vectorGrupo, vector<double>& vectorPromedio):
	tamGrupo = len(vectorGrupo)
	tamDato = len(vectorPromedio)
	anterior = vectorPromedio
	cambio = 1
    vectorPromedio.clear()
	vectorPromedio.resize(tamDato)
	for i in range(0,tamGrupo):
		for j in range(0, tamDato):
			vectorPromedio[j] += vectorGrupo[i][j]
	for i in range(0,tamDato):
		vectorPromedio[i] = vectorPromedio[i]//tamGrupo
	if compararVectores(anterior, vectorPromedio): #Fijo esto se puede optimizar con una libreria!!
		cambio = 0
	return cambio

def algoritmoLloyd(vector<vector<double>>& vectorDatos, vector<vector<double>>& vectorCentroides, puntosAsociadosC, m:int, k:int, eps:double) :
	posMin = 0
	fCosto = 0.0
	centroidesAlterados = 0
    vecsXproceso = m//size
    inicio = (pid*vecsXproceso)
    fin = inicio + vecsXproceso
	for i in range(inicio,fin):##Cuando le codigo este listo tal vez esto sea innecesario
		fCosto += calcDisMin(vectorDatos[i], vectorCentroides)
		posMin = calcMinPos(vectorDatos[i], vectorCentroides)
        puntosAsociadosC[posMin].append(i)	#FALTA VER COMO MANEJAR ESTO
	fCosto = comm.allreduce(fCosto, op=MPI.SUM)
    fCostoPrime = fCosto
	while True: #El equivalente a un do-while
		fCosto = fCostoPrime
		centroidesAlterados = 0
		for i in range(0,len(puntosAsociadosC)):#IDEA: if pid < k: promediarGrupo[pid]
			centroidesAlterados += promediarGrupo(puntosAsociadosC[i], vectorCentroides[i])
        #centroidesAlterados = comm.allreduce(centroidesAlterados, op=MPI.SUM)    
		puntosAsociadosC.clear()
		puntosAsociadosC.resize((k,0))
		fCostoPrime = 0
		for i in range(inicio,fin):
			fCostoPrime += calcDisMin(vectorDatos[i], vectorCentroides)
			posMin = calcMinPos(vectorDatos[i], vectorCentroides)
            puntosAsociadosC[posMin].append(i) #FALTA VER COMO MANEJAR ESTO
        fCostoPrime = comm.allreduce(fCostoPrime, op=MPI.SUM)
	if ((fCosto - fCostoPrime) <= eps) || (centroidesAlterados == 0): #El equivalente a un do-while
        break
	return fCostoPrime

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
    print(my_data," ",m," ",n," ",e)
    centroides = initParalelo(my_data, k, m, n)
    centroidesFinales = []
    l = m // 2
    l2 = l //size
    if pid == 0:
        for i in range(size):
            for j in range(l2):
                centroidesFinales.append(centroides[i][j])
        centroidesFinales = list(dict.fromkeys(centroidesFinales))
        print(centroidesFinales)

    return

if __name__ == "__main__":
    main(sys.argv[1:])