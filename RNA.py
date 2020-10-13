# importação das bibliotecas necessárias

# pybrain 
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# gráficos 
#import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data


# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( 45, 1000, 1000, 1 )    # define network 
dataSet = SupervisedDataSet( 45, 1 )  # define dataSet

'''
arquivos = ['1.txt', '1a.txt', '1b.txt', '1c.txt',
            '1d.txt', '1e.txt', '1f.txt']
'''  
arquivos = ['1.txt', '1b.txt', '1c.txt', '2.txt', '2b.txt', '2c.txt', '3.txt', '3b.txt', '3c.txt', '4.txt', '4b.txt', '4c.txt', '5.txt', '5b.txt', '5c.txt', '6.txt', '6b.txt', '6c.txt', '7.txt', 
            '7b.txt', '7c.txt', '8.txt', '8b.txt', '8c.txt', '9.txt', '9b.txt', '9c.txt', '0.txt', '0b.txt', '0c.txt' ]   
            
# a resposta do número
resposta = [ [1], [1], [1], [2], [2], [2], [3], [3], [3], [4], [4], [4], [5], [5], [5], [6], [6], [6], [7], [7], [7], [8], [8], [8], [9], [9], [9], [0], [0], [0] ]
#resposta = [[1], [1], [1], [1], [1], [1], [1]] 

i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getData( arquivo )            # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w") # arquivo para guardar os resultados

while error > 0.001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
arquivos = ['1- test.txt', '2- test.txt', '3- test.txt', '4- test.txt', '5- test.txt', '6- test.txt', '7- test.txt', '8- test.txt', 
            '9- test.txt', '0- test.txt']
            
for arquivo in arquivos:
    data =  getData( arquivo )
    print ( network.activate( data ) )


# plot graph
#plt.ioff()
#plt.plot( outputs )
#plt.xlabel('Iterações')
#plt.ylabel('Erro Quadrático')
#plt.show()

