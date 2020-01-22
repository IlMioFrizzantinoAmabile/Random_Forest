import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Random_Forest import *
import time
tic=time.time()


#import dataset
MNISTtest = pd.read_csv('MNISTtest.csv')
MNISTtrain = pd.read_csv('MNISTtrain.csv')
    
cifra_da_classificare = 7
for df in [MNISTtest,MNISTtrain]:
    for i in range(df.shape[0]):
        if df.iloc[i]['y']==cifra_da_classificare:
            df.iloc[i]['y']=1
        else:
            df.iloc[i]['y']=0  

   
    
percentuale_train_alberi=0.1
numero_alberi=100
numero_alberisss=[1,2,4,10,20,100]
numero_nodi=100



random_forest = Foresta('y', MNISTtrain, numero_alberi, percentuale_train_alberi)

errori_quad = {}
errori_perc = {}
errori_quad_train = {}
errori_perc_train = {}
for alb in numero_alberisss:
    errori_quad[str(alb)] = []
    errori_perc[str(alb)] = []
    errori_quad_train[str(alb)] = []
    errori_perc_train[str(alb)] = []
    
for num_nodo in range(0,numero_nodi):
    print('\n\n\n\n aggiungo nodo ',num_nodo+1)
    if num_nodo>0:
        random_forest.train(STAMPA=True)
    
    for alb in numero_alberisss:
        errori=[]
        indovinate=0
        for i in range(MNISTtest.shape[0]):     #TEST ERROR
            passeggero = MNISTtest.iloc[i]
            esito_voto = random_forest.guess(passeggero,alb)                #esito_voto è float in [0,1]
            esito_reale = passeggero[random_forest.predicted_feature]   #esito_reale è intero in {0,1}
            scarto = np.abs(esito_voto - esito_reale)
            errori.append(scarto)
            if np.abs(scarto) < 0.5:
                indovinate+=1
        err_quad = np.linalg.norm(errori,2)/np.sqrt(MNISTtest.shape[0])
        err_perc = 100*(MNISTtest.shape[0]-indovinate)/MNISTtest.shape[0]
        errori_quad[str(alb)].append(err_quad)
        errori_perc[str(alb)].append(err_perc)
        
        errori=[]
        indovinate=0
        for i in range(MNISTtrain.shape[0]):       #TRAINING ERROR
            passeggero=MNISTtrain.iloc[i]
            esito_voto = random_forest.guess(passeggero,alb)                #esito_voto è float in [0,1]
            esito_reale = passeggero[random_forest.predicted_feature]   #esito_reale è intero in {0,1}
            scarto = np.abs(esito_voto - esito_reale)
            errori.append(scarto)
            if np.abs(scarto) < 0.5:
                indovinate+=1
        err_quad = np.linalg.norm(errori,2)/np.sqrt(MNISTtrain.shape[0])
        err_perc = 100*(MNISTtrain.shape[0]-indovinate)/MNISTtrain.shape[0]
        errori_quad_train[str(alb)].append(err_quad)
        errori_perc_train[str(alb)].append(err_perc)
        
    eee='Errore perc: '
    for alb in numero_alberisss:
        eee += str(errori_perc[str(alb)][num_nodo])+' , '
    print(eee)

for alb in numero_alberisss:
    #plot train set errors in norm 2 variing the number of nodes
    f=plt.figure(1+10*alb)
    plt.plot(range(1,numero_nodi+1),errori_quad_train[str(alb)][:numero_nodi])
    plt.plot(range(1,numero_nodi+1),errori_quad[str(alb)][:numero_nodi])
    f.legend(['Training Set','Test Set'], loc='upper center', bbox_to_anchor=(0.5, 0.2, 0.5, 0.6))
    plt.xlabel('Numero nodi')
    plt.ylabel('Errore in norma 2')
    plt.title('Numero di alberi = '+str(alb))
    plt.savefig('MNISTplot\Alberi'+str(alb)+'a'+str(percentuale_train_alberi)+' - Err 2.png')
    
    
    #plot train and test set errors in % variing the number of nodes
    g=plt.figure(2+10*alb)
    plt.plot(range(1,numero_nodi+1),errori_perc_train[str(alb)][:numero_nodi])
    plt.plot(range(1,numero_nodi+1),errori_perc[str(alb)][:numero_nodi])
    g.legend(['Training Set','Test Set'], loc='upper center', bbox_to_anchor=(0.5, 0.2, 0.5, 0.6))
    plt.xlabel('Numero nodi')
    plt.ylabel('Errore in percentuale')
    plt.title('Numero di alberi = '+str(alb))
    plt.savefig('MNISTplot\Alberi'+str(alb)+'a'+str(percentuale_train_alberi)+' - Err perc.png')
   

tempoo = time.time() - tic
print('ci ho messo ',tempoo)

