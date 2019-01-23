import os
import pickle
from tqdm import tqdm
from ScalableNaiveBayes import *
from Bio import SeqIO
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report


def create_data(child_id):

    all_clusters = next(os.walk('../Ortholog/kegg/completed/all/Fasta'))[2]    
    my_clusters  = next(os.walk('./clusters/all250'))[2]   
    
    for clstr in tqdm(all_clusters):

        if clstr not in my_clusters:


            genes = list(SeqIO.parse('../Ortholog/kegg/completed/all/Fasta/'+clstr,'fasta'))    

            if(len(genes)>3000):    
            
                child = pickle.load(open(os.path.join('models','modelall5',str(child_id)+'.p'),'rb'))
                Y = child.predict(genes)

                pickle.dump(Y,open(os.path.join('anomalous_data',clstr.split('.')[0]+'_'+str(child_id)+'.p'),'wb'))
	

def load_data():

    all_clusters = next(os.walk('../Ortholog/kegg/completed/all/Fasta'))[2]    
    my_clusters  = next(os.walk('./clusters/all250'))[2]   

    

    for clstr in tqdm(my_clusters):
        

        clstr_probs = []

        for child in tqdm(range(1,16)):

            probs = pickle.load(open('./anomalous_data/'+clstr.split('.txt')[0]+'_'+str(child)+'.p','rb'))

            if(len(clstr_probs)==0):

                for i in range(len(probs)):

                    clstr_probs.append([])

            for gene in tqdm(range(len(probs))):

                for clsID in tqdm(probs[gene]):

                    if(clsID=='max' or clsID=='maxClass'):
                        continue

                    else:
                        clstr_probs[gene].append(probs[gene][clsID])

        for gene in range(len(clstr_probs)):

            clstr_probs[gene] = np.array(clstr_probs[gene])
            clstr_probs[gene] = clstr_probs[gene] - np.min(clstr_probs[gene])
            clstr_probs[gene] = clstr_probs[gene] / np.max(clstr_probs[gene])
            clstr_probs[gene].sort()

        pickle.dump(clstr_probs, open('./processed_data/good/'+clstr.split('.txt')[0]+'.p','wb'))



    for clstr in tqdm(all_clusters):

        if clstr not in my_clusters:

            clstr_probs = []

            try:

                for child in tqdm(range(1,16)):

                    probs = pickle.load(open('./anomalous_data/'+clstr.split('.txt')[0]+'_'+str(child)+'.p','rb'))

                    if(len(clstr_probs)==0):

                        for i in range(len(probs)):

                            clstr_probs.append([])

                    for gene in tqdm(range(len(probs))):

                        for clsID in tqdm(probs[gene]):

                            if(clsID=='max' or clsID=='maxClass'):
                                continue

                            else:
                                clstr_probs[gene].append(probs[gene][clsID])

                for gene in range(len(clstr_probs)):

                    clstr_probs[gene] = np.array(clstr_probs[gene])
                    clstr_probs[gene] = clstr_probs[gene] - np.min(clstr_probs[gene])
                    clstr_probs[gene] = clstr_probs[gene] / np.max(clstr_probs[gene])
                    clstr_probs[gene].sort()

                pickle.dump(clstr_probs, open('./processed_data/bad/'+clstr.split('.txt')[0]+'.p','wb'))

            except:

                pass

def feature_extraction():

    X = []
    Y = []

    clstrs = next(os.walk('./processed_data/good'))[2]

    for clstr in tqdm(clstrs,desc='Good'):

        probas = pickle.load(open('./processed_data/good/'+clstr,'rb'))

        for proba in tqdm(probas):

            mean = np.mean(proba)
            std = np.std(proba)
            summ = np.sum(proba)
            sk = skew(proba)
            kur = kurtosis(proba)

            x = [mean, std, summ, sk, kur]

            X.append(x)
            Y.append(1)



    clstrs = next(os.walk('./processed_data/bad'))[2]

    for clstr in tqdm(clstrs,desc='Bad'):

        probas = pickle.load(open('./processed_data/bad/'+clstr,'rb'))

        for proba in tqdm(probas):

            mean = np.mean(proba)
            std = np.std(proba)
            summ = np.sum(proba)
            sk = skew(proba)
            kur = kurtosis(proba)

            x = [mean, std, summ, sk, kur]

            X.append(x)
            Y.append(0)


    pickle.dump(Y,open('Y.p','wb'))
    Y = None
    pickle.dump(X,open('X.p','wb'))
    

def study_features():

    X = pickle.load(open('X.p','rb'))
    Y = pickle.load(open('Y.p','rb'))

    cols = ['mean', 'std', 'summ', 'sk', 'kur']

    for i in tqdm(range(5)):
        good = []
        bad = []

        for j in tqdm(range(len(X))):

            if(Y[j]==1):

                good.append(X[j][i])

            else:
                bad.append(X[j][i])

        a = np.histogram(good,bins=30,range=(min(min(good),min(bad)),max(max(good),max(bad))),density=True)[0]
        b = np.histogram(bad,bins=30,range=(min(min(good),min(bad)),max(max(good),max(bad))),density=True)[0]
            
        plt.hist([a, b], color=['b','r'], alpha=0.5)
        plt.title(cols[i])
        plt.savefig(cols[i]+'.png')
        plt.close() 


def RandomForestIndependent():

    X = pickle.load(open('X.p','rb'))
    Y = pickle.load(open('Y.p','rb'))

    print('****  *****')

    rf = RandomForestClassifier(n_estimators=10)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.0, random_state=3)

    rf.fit(X_train,Y_train)
    yp = rf.predict(X_train)
    print('**** Training *****')
    print(classification_report(Y_train, yp))

    '''yp = rf.predict(X_test)
    print('**** Testing *****')
    print(classification_report(Y_test, yp,digits=6))'''

    pickle.dump(rf, open('rf.p','wb'))
    

def stkFoldCrossValidation():

    X = pickle.load(open('X.p','rb'))

    X = np.array(X)

    Y = pickle.load(open('Y.p','rb'))

    Y = np.array(Y)

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, Y)
    
    k = 1
    for train_index, test_index in skf.split(X,Y):
   
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        print(k)
        k += 1

        rf = RandomForestClassifier()
        
        rf.fit(X_train,Y_train)

        yp = rf.predict(X_test)
        print(classification_report(Y_test, yp,digits=6))

    
def main():
    RandomForestIndependent()



if __name__ == '__main__':
    main()