import random

ben = [3, 3, 2, 4, 2, 3, 5, 2]
weight = [5, 4, 7, 8, 4, 4, 6, 8]

def getCromossomo( size ):
  cromossomo = []
  i = 0
  while i < size:
    gene = random.randint(0, 1)
    cromossomo.append(gene)
    i = i + 1
  return cromossomo

size = 8
cromossomo = getCromossomo( size )

def fitness( cromossomo ):
  i = 0
  beneficio = 0
  peso = 0
  while i < size:
    gene = cromossomo[i]
    if gene == 1:
      beneficio = beneficio + ben[i]
      peso = peso + weight[i]
    i = i + 1
  if peso > 25:
    beneficio = -1
  return beneficio      

beneficio = -1
sizePop = 10

while beneficio < 13:
  cromossomo = getCromossomo( size )
  beneficio = fitness( cromossomo ) 
  print(cromossomo)  
  print(beneficio)
