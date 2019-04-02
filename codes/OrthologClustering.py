from tqdm import tqdm
import pickle
from math import ceil
from ScalableNaiveBayes import ScalableNaiveBayes
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

def stratifiedKFoldCrossValidation(dataFile, K, n_clusters, id, n_child):
    '''
    Perform k-fold cross validation (stratified)
    
    Arguments:
        dataFile {str} -- name/path to pickle file containg data (generated using ReadFastas)
        K {int} -- value of k as in k-mers
        n_clusters {int} -- number of clusters under consideration
        id {str} -- model id
        n_child {int} -- number of child instances of Naive Bayes to use. 
                         higher number of n_child reduces memory requirement but is slower
    '''

    (X, Y, v_size) = pickle.load(open(dataFile, 'rb'))      # load the data

    data = {}               # gene sequences in a cluster
    cnt = {}                # count of genes in a cluster
    segments = {}           # for k-folding ( should've used sklearn instead :/ )

    results = []            # list to store the validation results
    results.append(('True', 'Predicted'))


    for i in tqdm(range(len(X))):

        if(Y[i] not in data):

            data[Y[i]] = []
            cnt[Y[i]] = 0

        data[Y[i]].append(X[i])
        cnt[Y[i]] += 1


    for className in tqdm(cnt, total=len(cnt)):

        tot = cnt[className]            # total number of genes in the cluster

        dt = ceil(tot/10)               # number of gene in each fold

        segments[className] = []        # segments in each fold

        for i in range(9):

            segments[className].append((dt*i, dt*(i+1)))

        segments[className].append((9*dt, tot))     # final fold

    X = 'garbage'        # literal garbage collection :p
    Y = 'garbage'        # literal garbage collection :p


    for k in tqdm(range(10)):

        XTrain = []
        YTrain = []

        XTest = []
        YTest = []

        for className in tqdm(data, total=len(data)):

            for i in tqdm(range(10)):

                for j in tqdm(range(segments[className][i][0], segments[className][i][1])):

                    if(i == k):

                        XTest.append(data[className][j])
                        YTest.append(className)

                    else:

                        XTrain.append(data[className][j])
                        YTrain.append(className)

        (documents, total) = formatData(XTrain, YTrain)       # format the data to be used for training

        XTrain = 'garbage'        # literal garbage collection :p
        YTrain = 'garbage'        # literal garbage collection :p

        nb = ScalableNaiveBayes(model_id=str(id)+str(k), v_size=v_size, total_data=total, n_child=min(n_child, n_clusters)) # initializing the naive bayes model

        nb.train(documents, K)      # training the naive bayes model

        documents = 'garbage'        # literal garbage collection :p

                

        yPred = nb.predict(XTest, proba=False)


        for sample in tqdm(range(len(XTest))):

            results.append((YTest[sample], yPred[sample]))

    pickle.dump(results, open('ST10FoldResults-'+str(id)+'.p', 'wb'))


def classificationPerformance(result_file):
    '''
    Observe the results of startified 10 fold cross validation
    
    Arguments:
        result_file {str} -- name / path of the computed result file
    '''
    
    results = pickle.load(open(result_file, 'rb'))

    YT = []     # true labels
    YP = []     # predicted labels

    for i in range(1, len(results)):        # first index is header
        YT.append(results[i][0])
        YP.append(results[i][1])


    print(classification_report(YT, YP, digits=4))

def plotConfusionMatrix(result_file):
    '''
    plot the confusion matrix of startified 10 fold cross validation
    
    Arguments:
        result_file {str} -- name / path of the computed result file
    '''

    results = pickle.load(open(result_file, 'rb'))

    YT = []     # true labels
    YP = []     # predicted labels

    for i in range(1, len(results)):        # first index is header
        YT.append(results[i][0])
        YP.append(results[i][1])


    cm = confusion_matrix(YT, YP)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = list(set(YT))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest',cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),           
           xticklabels=classes, yticklabels=classes,           
           ylabel='True cluster',
           xlabel='Predicted cluster')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")    
    fmt = '.4f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()


def formatData(X, Y):
    '''
    Formates the data to train the Naive Bayes model    
    
    Arguments:
        X {list} -- gene sequences
        Y {list} -- cluster labels
    
    Returns:
        [list] -- returns the formatted data as [ ((x11,x12,..,x1n),Y1), ((x21,x22,..,x2n),Y2) ... ] 
    '''

    documents = []
    clsMapper = {}
    vocabulary = {}

    total = len(X)    

    for i in tqdm(range(len(X))):

        try:

            documents[clsMapper[Y[i]]][0].append(X[i])

        except:

            clsMapper[Y[i]] = len(clsMapper)
            documents.append(([X[i]], Y[i]))


    return (documents, total)
