import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Random_Forest import *

#read dataset
all_titanic = pd.read_csv("titanic.csv")
all_titanic.drop(['Name'], axis='columns', inplace=True)


#best train and single test
if False:
    #parameter
    percentuale_train_alberi=0.7
    numero_alberi=10
    numero_nodi=20
    
    #train
    random_forest = Foresta('Survived', all_titanic, numero_alberi, percentuale_train_alberi)
    random_forest.train(numero_nodi)

    #test
    paolo = pd.DataFrame(
        {
         'Survived': [0],
         'Pclass':[1],
         'Sex': ['male'],
         'Age': [23],
         'Siblings/Spouses Aboard': [2],
         'Parents/Children Aboard': [1],
         'Fare': [23.23]
        }
    ).iloc[0]    
    print('\n\n Paolo morirà con probabilità: ',random_forest.guess(paolo))
    
    
    
    
    
#finding optimal parameter
if False:
    #training set (titanic) and test set (check_titanic)
    percentuale_train_foresta=0.8
    titanic = all_titanic.iloc[:int(percentuale_train_foresta*all_titanic.shape[0])]
    check_titanic = all_titanic.iloc[int(percentuale_train_foresta*all_titanic.shape[0]):]
        
    #parameter
    percentuale_train_alberi=0.5
    numero_alberi=100
    numero_nodi=100  #will check for all values from 1 to numero_nodi
    
    #inizialize forest
    random_forest = Foresta('Survived', titanic, numero_alberi, percentuale_train_alberi)
    
    #for each step trains one more node for each tree, then compute the error
    errori_quad = []
    errori_perc = []
    errori_quad_train = []
    errori_perc_train =[]
    for num_nodo in range(0,numero_nodi):
        print('\n aggiungo nodo ',num_nodo+1)
        if num_nodo>0:
            random_forest.train(STAMPA=True)  #STAMPA=True per vedere l'andamento del training
        
        errori=[]
        indovinate=0
        for i in range(check_titanic.shape[0]):     #TEST ERROR
            passeggero = check_titanic.iloc[i]
            esito_voto = random_forest.guess(passeggero)                #esito_voto è float in [0,1]
            esito_reale = passeggero[random_forest.predicted_feature]   #esito_reale è intero in {0,1}
            scarto = np.abs(esito_voto - esito_reale)
            errori.append(scarto)
            if np.abs(scarto) < 0.5:
                indovinate+=1
        err_quad = np.linalg.norm(errori,2)/np.sqrt(check_titanic.shape[0])
        err_perc = 100*(check_titanic.shape[0]-indovinate)/check_titanic.shape[0]
        errori_quad.append(err_quad)
        errori_perc.append(err_perc)
        
        errori=[]
        indovinate=0
        for i in range(titanic.shape[0]):       #TRAINING ERROR
            passeggero=titanic.iloc[i]
            esito_voto = random_forest.guess(passeggero)                #esito_voto è float in [0,1]
            esito_reale = passeggero[random_forest.predicted_feature]   #esito_reale è intero in {0,1}
            scarto = np.abs(esito_voto - esito_reale)
            errori.append(scarto)
            if np.abs(scarto) < 0.5:
                indovinate+=1
        err_quad = np.linalg.norm(errori,2)/np.sqrt(titanic.shape[0])
        err_perc = 100*(titanic.shape[0]-indovinate)/titanic.shape[0]
        errori_quad_train.append(err_quad)
        errori_perc_train.append(err_perc)
    

    #plot train set errors in norm 2 variing the number of nodes
    f=plt.figure(1)
    plt.plot(range(1,numero_nodi+1),errori_quad_train)
    plt.plot(range(1,numero_nodi+1),errori_quad)
    f.legend(['Training Set','Test Set'], loc='upper center', bbox_to_anchor=(0.5, 0.2, 0.5, 0.6))
    plt.xlabel('Numero nodi')
    plt.ylabel('Errore in norma 2')
    plt.title('Numero di alberi = '+str(numero_alberi))
    plt.savefig('TITANICplot\Alberi'+str(numero_alberi)+'a '+str(percentuale_train_alberi)+' - Err 2.png')
    
    
    #plot train and test set errors in % variing the number of nodes
    g=plt.figure(2)
    plt.plot(range(1,numero_nodi+1),errori_perc_train)
    plt.plot(range(1,numero_nodi+1),errori_perc)
    g.legend(['Training Set','Test Set'], loc='upper center', bbox_to_anchor=(0.5, 0.2, 0.5, 0.6))
    plt.xlabel('Numero nodi')
    plt.ylabel('Errore in percentuale')
    plt.title('Numero di alberi = '+str(numero_alberi))
    plt.savefig('TITANICplot\Alberi'+str(numero_alberi)+'a '+str(percentuale_train_alberi)+' - Err perc.png')
    