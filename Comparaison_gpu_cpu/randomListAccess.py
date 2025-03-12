from random import randint as rd
import time as tm

def createList(x:int, i:int, N:int) -> list:
    #On créer une liste aléatoire
    listeToAccess = [rd(0, 100) for i in range(N)]
    listeToAccess[i] = x
    return listeToAccess


def createDict(x:int, i:int, N:int) -> dict:
    #On créer un dictionnaire aléatoire
    dictToAccess = {i:rd(0, 100) for i in range(N)}
    dictToAccess[i] = x
    return dictToAccess

def getElementList(x:int, i:int, N:int) -> tuple:

    struct = createList(x, i, N)

    #Recherche naive
    for i in range(len(struct)):
        # if (i % 10000 == 0):
        #     print("Recherche en cours : ", i)
        if (struct[i] == x):
            return (i, x)
    return ('/', '/')


def getElementDict(x:int, i:int, N:int) -> tuple:

    struct = createDict(x, i, N)

    #Recherche dict
    for i in range(len(struct)):
        # if (i % 10000 == 0):
        #     print("Recherche en cours : ", i)
        if x in struct:
            return (struct[x], x)
    return ('/', '/')


if __name__ == "__main__":
    x = 8888
    i = 8233455
    N = 10000000

    start = tm.time()
    print(getElementList(x, i, N))
    end = tm.time()
    print("Recherche naive (LISTE) CPU de ", x, " sur ", N, "est de ", end - start, "s")

    start = tm.time()
    print(getElementDict(x, i, N))
    end = tm.time()
    print("Recherche naive (DICT) CPU de ", x, " sur ", N, "est de ", end - start, "s")
