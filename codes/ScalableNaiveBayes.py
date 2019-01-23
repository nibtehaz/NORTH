import os 
from math import log10
import pickle
from tqdm import tqdm
from FeatureExtraction import breakIntoKmer
from math import ceil
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sqlite3

#global NCHILD
#NCHILD = 3


np.random.seed(3)


class ScalableNaiveBayes(object):

    def __init__(self,modelName,vSize=0,totalData=0,alpha=1.0,nJobs=1,nChild=10):

        if(os.path.isdir(modelName)):

            pass

            #raise ValueError("A model named ",modelName," already exists in the directory")
            #return
        else :

            os.makedirs(modelName)

        self.modelName = modelName
        self.childs = []
        self.nChild = nChild
        self.alpha = alpha
        self.nJobs = nJobs
        self.vSize = vSize
        self.totalData = totalData

    def train(self,data,K,childsToEmploy='All'):

        if(childsToEmploy=='All'):

            childsToEmploy = self.nChild

        classesPerChild = ceil(len(data) / childsToEmploy)


        #print('Training')

        for i in tqdm(range(1,childsToEmploy+1)):

            dataPartition = data[(i-1)*classesPerChild : min(i*classesPerChild,len(data))]

            #print(dataPartition)

            child = NaiveBayesChild(str(i),self.vSize,self.alpha,K)
            child.train(dataPartition,self.totalData)
            pickle.dump(child,open(self.modelName+'/'+str(i)+'.p','wb'))
            self.childs.append(str(i)+'.p')


        # last child 

        #dataPartition = data[(childsToEmploy-1)*classesPerChild : len(data)]

        #child = NaiveBayesChild(str(childsToEmploy),self.vSize,self.alpha)
        #child.train(dataPartition,self.totalData)
        #pickle.dump(child,open(self.modelName+'/'+str(childsToEmploy)+'.p','wb'))
        #self.childs.append(str(childsToEmploy)+'.p')

        pickle.dump(self,open(self.modelName+'/model.p','wb'))

    '''def train(self,data,childsToEmploy='All'):

        if(childsToEmploy=='All'):

            childsToEmploy = self.nChild

        classesPerChild = ceil(len(data) / childsToEmploy)


        #print('Training')

        for i in tqdm(range(1,childsToEmploy+1)):

            dataPartition = data[(i-1)*classesPerChild : min(i*classesPerChild,len(data))]

            #print(dataPartition)

            child = NaiveBayesChild(str(i),self.vSize,self.alpha)
            child.train(dataPartition,self.totalData)
            pickle.dump(child,open(self.modelName+'/'+str(i)+'.p','wb'))
            self.childs.append(str(i)+'.p')


        # last child 

        #dataPartition = data[(childsToEmploy-1)*classesPerChild : len(data)]

        #child = NaiveBayesChild(str(childsToEmploy),self.vSize,self.alpha)
        #child.train(dataPartition,self.totalData)
        #pickle.dump(child,open(self.modelName+'/'+str(childsToEmploy)+'.p','wb'))
        #self.childs.append(str(childsToEmploy)+'.p')

        pickle.dump(self,open(self.modelName+'/model.p','wb'))'''


    def predict(self,X,id='0',proba=True):

        '''
            Input : X is a list or numpy array
        '''
        
        if not (os.path.isdir(self.modelName+'/'+id)):

            os.makedirs(self.modelName+'/'+id)
            fp = open(self.modelName+'/'+id+'/log.txt','w')
            fp.close()
            pickle.dump(None,open(self.modelName+'/'+id+"/Y.p","wb"))


        fp = open(self.modelName+'/'+id+'/log.txt','r')
        logTxt=fp.readlines()
        fp.close()

        cmpltedChild = len(logTxt)
        ##########################################
        cmpltedChild = 0 #########################
        ##########################################
        Y=pickle.load(open(self.modelName+'/'+id+"/Y.p","rb"))


        for childID in tqdm(range(cmpltedChild,self.nChild)):
            child = None 
            child = pickle.load(open(self.modelName+'/'+self.childs[childID],"rb")) 

            res = child.predict(X)


            if Y==None:

                Y = res[:]

            else:

                for i in range(len(Y)):

                    if(Y[i][0]<res[i][0]):

                        Y[i][0] = res[i][0]
                        Y[i][1] = res[i][1]


            pickle.dump(Y,open(self.modelName+'/'+id+"/Y.p","wb"))
            
            fp = open(self.modelName+'/'+id+'/log.txt','a')
            logTxt=fp.write('Child '+str(childID+1)+' Completed'+'\n')
            fp.close()

        
        if not proba:

            for i in tqdm(range(len(Y))):

                Y[i] = Y[i][1]

        

        return Y[:]

    def childwisePredict(self,X,childID,proba=True):

        '''
            Input : X is a list or numpy array
        '''

        child = pickle.load(open(self.modelName+'/'+self.childs[childID],"rb")) 

        Y = child.predict(X)

        pickle.dump(Y,open(self.modelName+'/Y.p',"wb"))
        
        if not proba:

            for i in tqdm(range(len(Y))):

                Y[i] = Y[i][1]

        return Y[:]


class NaiveBayesChild(object):

    def __init__(self,ID,vSize,alpha,K):

        self.classes = []
        self.prior = {}
        self.probabilities = {} 
        self.defaultRatio = {}
        self.alpha = alpha
        self.id = ID 
        self.vSize = vSize
        self.K = K

    def train(self, data, totalData):

        #print(len(data))
        #print(len(data[0]))

        #print(data)

        for docClass in tqdm(data,total=len(data)):

            #tqdm.write(str(len(docClass)))
            #tqdm.write(str(len(docClass[0])))


            classLabel = docClass[1]

            self.classes.append(classLabel)
            #print(self.classes)

            self.prior[classLabel] = log10(len(data[0]) / totalData) # to avoid recomputing log10 values again and again 
            self.probabilities[classLabel] = {}

            totalWords = 0


            for text in tqdm(docClass[0],total=len(docClass[0])):

                tokenizedText = breakIntoKmer(text,self.K).split(' ')
                #print(tokenizedText)

                

                for word in tqdm(tokenizedText,total=len(tokenizedText)):

                    #print(word)

                    totalWords += 1 

                    try:
                        
                        self.probabilities[classLabel][word] += 1 

                    except:  # word is not present

                        self.probabilities[classLabel][word] = 1 


            for word in tqdm(self.probabilities[classLabel],total=len(self.probabilities[classLabel])):

                self.probabilities[classLabel][word] = log10((self.probabilities[classLabel][word] + self.alpha) / (totalWords + self.alpha*self.vSize))  # to avoid recomputing log10 values again and again 

            self.defaultRatio[classLabel] = log10((self.alpha) / (totalWords + self.alpha*self.vSize))  # to avoid recomputing log10 values again and again 

    def predict(self, X):

        '''
            Input : list or numpy array of texts
            Output : predicts the output classeses in the following format
                        [ (classLabel, probability) ... for x in X ]

        '''

        #for cls in self.classes:

            #print(cls)

            #print(self.probabilities[cls])

        #tqdm.write('Classes:')

        #for cls in self.classes:
         #   tqdm.write(str(cls))

        #print('Classes:')

        #for cls in self.classes:
         #   print(str(cls))

        #t=input('')

        Y = []

        try:

            for i in tqdm(range(len(X))):

                x =  breakIntoKmer(X[i],self.K).split(' ')

                maxx = None 
                y = ''

                for className in tqdm(self.classes,total=len(self.classes)):

                    prediction = self.prior[className]

                    for word in tqdm(x,total=len(x)):

                        try:

                            prediction += self.probabilities[className][word]

                        except:

                            prediction += self.defaultRatio[className]

                        #tqdm.write(str(prediction))

                    if(maxx==None or prediction>maxx):

                        maxx = prediction
                        y = className
                
                Y.append([maxx,y])
                #t=input('')
            #for yy in Y:

                #print(str(yy[0])+' - '+str(yy[1]))


            return Y


        except Exception as e:

            print(e)

            print('Pass as input list or numpy array of texts')


def kFoldCrossValidation(dataFile,K,nClasses,id):


    (X,Y, vSize)=pickle.load(open(dataFile,'rb'))

    data = {}
    cnt = {}
    segments = {}
    vSize = 130343616

    results = []
    results.append(('True','Predicted'))

    print('Reading Data')

    for i in tqdm(range(len(X))):

        if(Y[i] not in data):

            data[Y[i]] = []
            cnt[Y[i]] = 0

        data[Y[i]].append(X[i])
        cnt[Y[i]] += 1

    print('Processing Classes')

    for className in tqdm(cnt,total=len(cnt)):

        tot = cnt[className]

        dt = ceil(tot/10)

        segments[className] = []

        for i in range(9):

            segments[className].append((dt*i,dt*(i+1)))

        segments[className].append((9*dt,tot))	

    X = None
    Y = None

    print("Cross Validating")

    for k in tqdm(range(10)):

        XTrain = []
        YTrain = []

        XTest = []
        YTest = []


        for className in tqdm(data,total=len(data)):

            for i in tqdm(range(10)):

                for j in tqdm(range(segments[className][i][0],segments[className][i][1])):

                    if(i==k):

                        XTest.append(data[className][j])
                        YTest.append(className)

                    else:

                        XTrain.append(data[className][j])
                        YTrain.append(className)

        
        (documents, total) = readData(XTrain,YTrain)

        XTrain = None
        YTrain = None

        nb = ScalableNaiveBayes(modelName='models/model'+str(id)+str(K),vSize=vSize,totalData=total)
        
        print('Training')

        nb.train(documents,K)

        documents = None

        good = 0 
        bad = 0

        print('Testing')

        yPred = nb.predict(XTest,id=str(k)+'-'+str(nClasses)+'-'+str(K),proba=False)

        #print('YTest ',len(YTest),' YPred ', len(yPred))
            
        for sample in tqdm(range(len(XTest))):
			
            results.append((YTest[sample],yPred[sample]))

            ##print(YTest[sample],yPred[sample])

            if(YTest[sample]==yPred[sample]):

                good += 1 

            else:
                
                bad += 1 

        tqdm.write(str(k)+' : Good = '+str(good)+' Bad = '+str(bad))
        
        pickle.dump(results,open('10FoldResults/'+str(k)+'.p','wb'))


    YT=[]
    YP=[]

    for i in range(1,len(results)):
        YT.append(results[i][0])
        YP.append(results[i][1])

    fp = open('results/'+ dataFile[:-2] +str(nClasses)+'K'+str(K)+'.txt','w')
    fp.write(classification_report(YT,YP,digits=4))
    fp.close()

    print(classification_report(YT,YP,digits=4))

    pickle.dump(confusion_matrix(YT,YP),open('cnfMat.p','wb'))

def benchmarkSimulation(mode,mode2,childID,benchmarkData='benchmarkProteins1104.p'):

    if(mode=='Train'):

        if(mode2=='read'):

            (X,Y,vSize)=pickle.load(open('3merAll5.p','rb'))
            data = {}
            cnt = {}

            print('Reading Data')

            for i in tqdm(range(len(X))):

                if(Y[i] not in data):

                    data[Y[i]] = []
                    cnt[Y[i]] = 0

                data[Y[i]].append(X[i])
                cnt[Y[i]] += 1

            (documents, total) = readData(X,Y)


            X = 'garbage'
            Y = 'garbage'


        elif(mode2=='load'):

            (documents, total) = pickle.load(open('data.p','rb'))
            (X,Y,vSize)=pickle.load(open('3merAll5.p','rb'))
            X = 'garbage'
            Y = 'garbage'


        nb = ScalableNaiveBayes(nChild=NCHILD,modelName='models/model',vSize=vSize,totalData=total)
        
        print('Training')

        nb.train(documents)


    elif(mode=="Evaluate"):

        if(mode2=='All'):

            nb = pickle.load(open('models/model/model.p','rb'))

            testData = pickle.load(open(benchmarkData,'rb'))

            results = {}

            conn = sqlite3.connect('benchmarkResults'+str(childID)+'.sqlite')
            c = conn.cursor()

            try:

                c.execute('CREATE TABLE Results (nIndex INTEGER, pID VARCHAR(100), predClass VARCHAR(100), probability VARCHAR(100))')

            except :

                pass
            
            st=0

            huge = {}
            huge[48806]=11752763
            huge[75171]=11253691
            huge[360312]=11443516
            huge[387295]=11504723

            huge[425604]="'"
            huge[427130]="'"

            x = []

            for index in tqdm(range(st,len(testData))):

                if index in huge:
                    continue

                sample = testData[index]

                #tqdm.write(str(index)+' '+str(len(sample[1])))

                pID = sample[0][:]

                x.append(sample[1])

            yPred = nb.childwisePredict(x,childID)

            pickle.dump(open('yPred'+str(childID)+'.p','wb'))

                #tqdm.write(str(yPred[0][1])+" , "+str(yPred[0][0]))

                #print("INSERT INTO Results (nIndex, pID, outClass) VALUES ('"+str(index)+"','"+str(pID)+"','"+str(yPred)+"')")

                #c.execute("INSERT INTO Results (nIndex, pID, predClass, probability) VALUES ('"+str(index)+"','"+str(pID)+"','"+str(yPred[0][1])+"','"+str(yPred[0][0])+"')")
                #conn.commit()

        if(mode2=='Single'):

            nb = pickle.load(open('models/model/model.p','rb'))

            testData = pickle.load(open(benchmarkData,'rb'))

            results = {}

            
            st=0

            huge = {}
            huge[48806]=11752763
            huge[75171]=11253691
            huge[360312]=11443516
            huge[387295]=11504723

            huge[425604]="'"
            huge[427130]="'"

            x = []

            for index in tqdm(range(st,len(testData))):

                if index in huge:

                    continue

                sample = testData[index]

                #tqdm.write(str(index)+' '+str(len(sample[1])))

                pID = sample[0][:]

                x=[sample[1]]

                yPred = nb.childwisePredict(x,childID)

                results[pID] = yPred[0]

                tqdm.write(str(len(results)))

                if(index%100==0):

                    pickle.dump(results,open('yPredChild'+str(childID)+'.p','wb'))

            pickle.dump(results,open('yPredChild'+str(childID)+'.p','wb'))


        if(mode2=='Minibatch'):

            nb = pickle.load(open('models/model/model.p','rb'))

            testData = pickle.load(open(benchmarkData,'rb'))

            try:
                results = pickle.load(open('models/model/yPredChild'+str(childID)+'.p','rb'))
                tqdm.write('Loaded previous result with '+str(len(results))+' predictions')

            except:
                results = {}
                tqdm.write('Starting from scrach')
            
            st=len(results)

            huge = {}
            huge[48806]=11752763
            huge[75171]=11253691
            huge[360312]=11443516
            huge[387295]=11504723

    


            x = []
            pID = []

            print(len(testData))

            for index in tqdm(range(st,len(testData))):

                if index in huge or len(testData[index][1])>100000:
                    results[testData[index][0][:]] = 'LATER'
                    continue

                sample = testData[index]

                #tqdm.write(str(index)+' '+str(len(sample[1])))

                pID.append(sample[0][:])

                x.append(sample[1])

                if(index%1000==0):

                    yPred = nb.childwisePredict(x,childID)

                    for saveI in range(len(pID)):

                        results[pID[saveI]] = yPred[saveI]

                    pID = []
                    x = []

                    pickle.dump(results,open('models/model/yPredChild'+str(childID)+'.p','wb'))

            print(len(x))

            if(len(x)!=0):
                yPred = nb.childwisePredict(x,childID)

                for saveI in range(len(pID)):

                    results[pID[saveI]] = yPred[saveI]

                pID = []
                x = []

                pickle.dump(results,open('models/model/yPredChild'+str(childID)+'.p','wb'))

            x = []
            y = []

            for index in tqdm(range(len(testData))):

                if index in huge or len(testData[index][1])>100000:
                    results[testData[index][0][:]] = 'LATER'
                    continue

                sample = testData[index]

                #tqdm.write(str(index)+' '+str(len(sample[1])))

                curPID = sample[0][:]

                if curPID not in results:

                    pID.append(sample[0][:])

                    x.append(sample[1])
            
            if(len(x)!=0):
            
                yPred = nb.childwisePredict(x,childID)

                for saveI in range(len(pID)):

                    results[pID[saveI]] = yPred[saveI]

                pID = []
                x = []

                pickle.dump(results,open('models/model/yPredChild'+str(childID)+'.p','wb'))
                print('saved')



          

    elif mode=='Write':

        out = ''

        for cls in tqdm(results,total=len(results)):

            for i in range(len(results[cls])):

                for j in range(i+1,len(results[cls])):

                        out += results[cls][i] + '\t' + results[cls][j] + '\n'


        fp = open('benchmarkPer.txt','w')
    
        fp.write(out)

        fp.close()






def readData(X,Y):

    '''
            Takes [ (X,Y) ... (X,Y)] as input
            Returns [ ((x1,x2,..,xn),Y) ... ] as output

    '''

    documents = []
    clsMapper = {}
    vocabulary = {}

    total = len(X)

    tqdm.write('Preparing for training')

    for i in tqdm(range(len(X))):

        #x = breakIntoKmer(X[i],K).split(' ')
        


        try:

            documents[clsMapper[Y[i]]][0].append(X[i])

        except:

            clsMapper[Y[i]] = len(clsMapper)
            documents.append(([X[i]],Y[i]))

    
    #print(vocabulary)
    #t=input('')


    pickle.dump((documents,total),open('data.p','wb'))

    #print(documents)
    #input('docs : '+str(len(documents)))

    return (documents, total)


def main():

   
    #id = input('id = ')
    #K = int(id)
    K=5
    #1FoldCrossValidation(id)

    childID = int(input('Child ID = '))

    #for childID in range(22,30):
    benchmarkSimulation(mode="Evaluate",mode2='Minibatch',childID=childID)


if __name__ == '__main__':

    main()
    






