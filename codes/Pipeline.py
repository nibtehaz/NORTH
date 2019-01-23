from ReadFastas import rawReadFastas
from ScalableNaiveBayes import kFoldCrossValidation
import os



def main():
    
    
    K = int(input('K = '))
    family = input('Family = ')
    nClasses = int(input('nClasses = '))

    
    Ks = [3,4,5,6]
    nClassesList = [50,100,150,200,250]
    
    
    for nClasses in nClassesList:
        for K in Ks:

            if(os.path.exists('results/'+ family +str(nClasses)+'K'+str(K)+'.txt')):

                continue

            else:
                print('K = '+str(K)+' nClasses = '+str(nClasses))
                rawReadFastas('completed/'+family,saveFile=family+'.p',nClasses=nClasses,minCnt=None,K=K)
                kFoldCrossValidation(family+'.p',K,nClasses,family)

if __name__ == '__main__':
    main()
