import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_gini(num_zeri,num_uni):
    num=num_zeri+num_uni
    if num==0:
        return 0
    gini = 1 - (num_zeri/num)**2 - (num_uni/num)**2
    return gini

class Nodo:
    def __init__(self,direzione=None,x=None):
        self.direzione=direzione 
        self.x=x
        self.foglia=True
        self.figlioMAG=None
        self.figlioMIN=None
        self.points = None
        self.num_points=None
        self.GUESS = None
        
        #attributi per memorizzare lo split ottimale e non doverlo ricalcolare
        self.split_gini_gain= None
        self.split_direzione= None
        self.split_x= None
        self.split_point= None
    def stampa(self,lev):
        space=lev*'  '
        if self.foglia:
            print(space+'('+str(lev)+') - Size='+str(self.num_points),'    guessa',self.GUESS)
        else:
            print(space+'('+str(lev)+') - Size=',self.num_points,',  splitta con',self.direzione,self.x)
        
        if self.figlioMAG!=None:
            self.figlioMAG.stampa(lev+1)
        if self.figlioMIN!=None:
            self.figlioMIN.stampa(lev+1)
    def MAG(self):
        if self.figlioMAG == None:
            self.figlioMAG = Nodo()
        return self.figlioMAG
    def MIN(self):
        if self.figlioMIN == None:
            self.figlioMIN = Nodo()
        return self.figlioMIN
    def split_ottimale(self,predicted_feature): 
        points=self.points
        direzioni = points.columns.drop(predicted_feature) #tolgo 'Survived' tra le direzioni tra cui splittare
        dimensione = points.shape[0]
        num_zeri = points[points[predicted_feature]==0].shape[0]
        num_uni = dimensione - num_zeri
        dad_gini = get_gini(num_zeri,num_uni)
        if num_zeri==0 or num_uni==0:
            return dad_gini,dad_gini,None,None,None
        opt_gini = dad_gini
        opt_direzione = None
        opt_x = None
        for direzione in direzioni: #guarda gli split per ogni attributo
            s_points = points.sort_values(direzione)
            points_value_predicted_f = s_points[predicted_feature]
            points_value_direzione = s_points[direzione]
            num_zeriMIN = 0
            num_uniMIN = 0
            i=0
            while i<dimensione-1: #guarda gli split per ogni valore
                #x = s_points.iloc[i][direzione]
                x = points_value_direzione.iloc[i]
                j=0
                #while (i+j<dimensione) and (x == s_points.iloc[i+j][direzione]):
                #    if s_points.iloc[i+j][predicted_feature]==0:
                while (i+j<dimensione) and (x == points_value_direzione.iloc[i+j]):
                    if points_value_predicted_f.iloc[i+j]==0:
                        num_zeriMIN+=1
                    else:
                        num_uniMIN+=1
                    j+=1
                i+=j-1
                giniMIN = get_gini(num_zeriMIN,num_uniMIN)
                giniMAG = get_gini(num_zeri - num_zeriMIN , num_uni - num_uniMIN)
                #media pesata dei gini tra figlio 'maggiore' e figlio 'minore'
                gini = ( giniMIN*(i+1) + giniMAG*(dimensione-i-1) ) / dimensione
                if gini < opt_gini:
                    opt_gini = gini
                    opt_direzione = direzione
                    opt_x = s_points.iloc[i][direzione]
                    opt_point = s_points.iloc[i]
                i+=1
        if opt_gini<dad_gini:
            self.split_gini_gain=dad_gini-opt_gini
            self.split_direzione=opt_direzione
            self.split_x=opt_x
            self.split_point=opt_point
            return dad_gini,opt_gini, opt_direzione, opt_x, opt_point
        elif opt_gini==dad_gini:
            #print('foglia non migliorabile',opt_gini)
            return dad_gini,opt_gini, None, None, None
        else:
            #non dovrebbe mai succedere
            return dad_gini,opt_gini, None, None, None
    def guess(self,predicted_feature):
        num_uni = self.points[self.points[predicted_feature]==1].shape[0]
        num_zeri = self.points[self.points[predicted_feature]==0].shape[0]
        self.GUESS = num_uni/(num_uni+num_zeri)
        return self.GUESS
class Albero:
    def __init__(self,predicted_feature,points):
        self.dimensione=1
        self.radice=Nodo()
        self.radice.points=points
        self.radice.num_points = points.shape[0]
        self.radice.guess(predicted_feature)
        self.predicted_feature = predicted_feature
        self.nodi = [self.radice]
        num_zeri = points[points[self.predicted_feature]==0].shape[0]
        num_uni = points[points[self.predicted_feature]==1].shape[0]
        self.gini = get_gini(num_zeri,num_uni)*points.shape[0]
    def foglie(self):
        foglie=[]
        for nodo in self.nodi:
            if nodo.foglia==True:
                foglie.append(nodo)
        return foglie
    def stampa(self):
        self.radice.stampa(0)
    def aggiungi(self,point,new_direzione,new_x):
        nodo=self.radice
        points = nodo.points
        while nodo.x != None:
            points = nodo.points
            if point[nodo.direzione] > nodo.x:
                points = points[points[nodo.direzione] > nodo.x]
                nodo = nodo.MAG()
            else:
                points = points[points[nodo.direzione] <= nodo.x]
                nodo = nodo.MIN()
        self.dimensione +=1
        nodo.x = new_x
        nodo.direzione = new_direzione
        nodo.foglia = False
        figlioMAG = nodo.MAG()
        figlioMAG.points = points[points[new_direzione] > new_x]
        figlioMAG.num_points = figlioMAG.points.shape[0]
        figlioMAG.guess(self.predicted_feature)
        self.nodi.append(figlioMAG)
        figlioMIN = nodo.MIN()
        figlioMIN.points = points[points[new_direzione] <= new_x]
        figlioMIN.num_points = figlioMIN.points.shape[0]
        figlioMIN.guess(self.predicted_feature)
        self.nodi.append(figlioMIN)
    def train(self,numero_split,STAMPA=False):
        for bau in range(numero_split):
            if self.gini>1e-13:     #altrimenti non è migliorabile  (l'errore è dato dalle sottrazioni iterate)
                if STAMPA:
                    print('Gini=',self.gini)
                foglie = self.foglie()
                opt_miglioramento=0
                for foglia in foglie:
                    if foglia.split_direzione==None:
                        dad_gini,son_gini,direzione,x,point = foglia.split_ottimale(self.predicted_feature)
                        gini_gain = dad_gini-son_gini
                    else:
                        gini_gain = foglia.split_gini_gain
                        direzione = foglia.split_direzione
                        x = foglia.split_x
                        point = foglia.split_point
                    #il miglioramento è dato dall'incremento di gini pesato per il numero di punti in quella sezione
                    miglioramento = gini_gain*foglia.num_points
                    if miglioramento > opt_miglioramento:
                        opt_miglioramento=miglioramento
                        opt_direzione=direzione
                        opt_x=x
                        opt_point=point
                self.gini -= opt_miglioramento
                if opt_miglioramento>0:
                    self.aggiungi(opt_point,opt_direzione,opt_x)
                    if STAMPA:
                        print('   miglioramento di ',opt_miglioramento,' con ',opt_direzione,opt_x)       
            else:
                self.gini=0.0
    def guess(self,point):
        nodo=self.radice
        while nodo.x!=None:
            if point[nodo.direzione] > nodo.x:
                nodo=nodo.figlioMAG
            else:
                nodo=nodo.figlioMIN
        return nodo.GUESS                  #versione con voto continuo
        #return int(nodo.GUESS+0.5)          #versione con voto discreto
class Foresta:
    def __init__(self,predicted_feature,points,numero_alberi,fraz_conoscienza_albero=0.7):
        self.alberi=[]
        self.dimensione = numero_alberi
        self.dimensione_alberi = 1
        self.predicted_feature = predicted_feature
        for uuu in range(numero_alberi):
            num_points = int(fraz_conoscienza_albero*points.shape[0])
            small_points=points.sample(frac=1).iloc[:num_points].copy(deep=True)
            #print(small_points)
            self.alberi.append(Albero(predicted_feature,small_points))
    def train(self,num_split=1,STAMPA=False):
        self.dimensione_alberi += num_split
        if STAMPA:
            print(' START TRAINING')
            i=1
        for albero in self.alberi:
            if STAMPA:
                print(' Albero ',i)
                i+=1
            albero.train(num_split, STAMPA)
    def guess(self,point,numero_votanti=1000000):
        if numero_votanti>self.dimensione:
            numero_votanti=self.dimensione
            print('non ci sono abbastanza alberi')
        voti=0
        for i in range(numero_votanti):
            voti += self.alberi[i].guess(point)
        voto=voti/numero_votanti
        return voto