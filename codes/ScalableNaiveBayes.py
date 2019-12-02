import os
import pickle
from tqdm import tqdm
from FeatureExtraction import breakIntoKmer
from math import ceil, log10
import numpy as np
import time
np.random.seed(3)


class ScalableNaiveBayes(object):
    '''
    The Scalable Naive Bayes class
    '''

    def __init__(self, model_id, v_size=0, total_data=0, alpha=1.0, n_jobs=1, n_child=10):
        '''
        Constructor for the Scalable Naive Bayes class

        Arguments:
            model_id {str} -- unique id of the model

        Keyword Arguments:
            v_size {int} -- size of vocabulary (default: {0})
            total_data {int} -- [description] (default: {0})
            alpha {float} -- for Naive Bayes smoothing (default: {1.0} => Laplace Smoothing)
            n_jobs {int} -- number of jobs to run (default: {1})
            n_child {int} -- number of child instances (default: {10})

        Raises:
            ValueError -- if already a model exists with same id
        '''        

        if(os.path.isdir(model_id)):

            raise ValueError("A model named ",model_id," already exists in the directory")
            return

        else:

            os.makedirs(model_id)

        self.model_id = model_id
        self.childs = []            # list of child instances
        self.n_child = n_child
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.v_size = v_size
        self.total_data = total_data



    def train(self, data, K, childs_to_employ='All'):
        '''
        Train the Naive Bayes Model
        
        Arguments:
            data {[type]} -- Training data
            K {int} -- value of k as in k-mers
        
        Keyword Arguments:
            childs_to_employ {int} -- how many child to use (default: {'All'})
        '''

        if(childs_to_employ == 'All'):      # use all the child(ren)

            childs_to_employ = self.n_child


        classes_per_child = ceil(len(data) / childs_to_employ)        # number of clusters per child


        for i in tqdm(range(1, childs_to_employ+1)):
            

            data_partition = data[(i-1)*classes_per_child: min(i*classes_per_child, len(data))]

            child = NaiveBayesChild(self.v_size, self.alpha, K)     # create a child instance
            child.train(data_partition, self.total_data)                    # train the child instance
            pickle.dump(child, open(os.path.join(self.model_id,str(i)+'.p'), 'wb'))   # save the trained child instance 
            self.childs.append(str(i)+'.p')             # update the child information of the mother model

        
        pickle.dump(self, open(os.path.join(self.model_id,'model.p'), 'wb'))     # save the updated Naive Bayes Model


    def predict(self, X, proba=True):
        '''
        Predicts the cluster of a *list* of gene sequences

        The user must provide a list, not a sequence
        (a list of a single sequence is okay too)

        Arguments:
            X {list or numpy array} -- the list of gene sequences
            
        
        Keyword Arguments:            
            proba {bool} -- return the probability or not (default: {True})
        
        Returns:
            [list] -- list of predicted clusters
        '''

        
        Y = None

        for childID in tqdm(range(0, self.n_child)):
            
            
            child = pickle.load(open(os.path.join(self.model_id,self.childs[childID]), "rb"))   # load the child instance

            res = child.predict(X)          # obtain predictions from child instance

            child = 'Garbage'       # literal garbage collection :p 

            if Y == None:           # first iteration

                Y = res[:]

            else:   
                
                for i in range(len(Y)):

                    if(Y[i][0] < res[i][0]):    # better probability ?

                        Y[i][0] = res[i][0]         # update the probability
                        Y[i][1] = res[i][1]         # update the cluster name

        if not proba:           # don't return the probability, cluster labels only

            for i in tqdm(range(len(Y))):
                

                Y[i] = Y[i][1]

        return Y[:]




class NaiveBayesChild(object):
    '''
    The Naive Bayes child instance class
    '''


    def __init__(self, v_size, alpha, K):
        '''
        Constructor for the Naive Bayes child instance class
        
        Arguments:
            v_size {int} -- size of vocabulary
            alpha {float} -- for Naive Bayes smoothing (default: {1.0} => Laplace Smoothing)
            K {int} -- value of k as in k-mers
        '''

        self.classes = []           # list of clusters
        self.prior = {}             # prior probabilities
        self.probabilities = {}     # probabilities of individual kmers
        self.defaultRatio = {}      # default probability for an unknown kmer
        self.alpha = alpha          
        self.v_size = v_size
        self.K = K  

    def train(self, data, total_data):
        '''
        Train the child instance
        
        Arguments:
            data {list} -- Training data
            total_data {int} -- Total number of data points
        '''

        for docClass in tqdm(data, total=len(data)):
            

            cluster_label = docClass[1]            # label of current cluster

            self.classes.append(cluster_label)      # append to list of clusters
            
                                # to avoid recomputing log10 values again and again
            self.prior[cluster_label] = log10(len(data[0]) / total_data)
            
            self.probabilities[cluster_label] = {}      # initializing k-mer probabilities

            total_kmers = 0

            for sequence in tqdm(docClass[0], total=len(docClass[0])):  
                

                tokenized_sequence = breakIntoKmer(sequence, self.K).split(' ')      # break the sequence into k-mers
                
                for kmer_ in tqdm(tokenized_sequence, total=len(tokenized_sequence)):
                    

                    total_kmers += 1

                    try:

                        self.probabilities[cluster_label][kmer_] += 1

                    except:  # kmer is not present

                        self.probabilities[cluster_label][kmer_] = 1
                                                                                                

                                            # to avoid recomputing log10 values again and again
            for kmer_ in tqdm(self.probabilities[cluster_label], total=len(self.probabilities[cluster_label])):
                

                self.probabilities[cluster_label][kmer_] = log10((self.probabilities[cluster_label][kmer_] + self.alpha) / (total_kmers + self.alpha*self.v_size))  
                                                    

                                            # to avoid recomputing log10 values again and again
            self.defaultRatio[cluster_label] = log10((self.alpha) / (total_kmers + self.alpha*self.v_size))

    def predict(self, X):
        '''
        Perform prediction using the child instance
        
        Arguments:
            X {list or numpy array} -- the list of gene sequences
        
        Returns:
            [list] -- predicts the output clusters in the following format
                        [ (cluster_label, probability) ... for x in X ]
        '''


        Y = []      # output clusters

        try:

            for i in tqdm(range(len(X))):
                

                x = breakIntoKmer(X[i], self.K).split(' ')      # break the sequence into k-mers

                maxx = None         # initialization
                y = ''

                for cluster_name in tqdm(self.classes, total=len(self.classes)):
                    

                    prediction = self.prior[cluster_name]

                    for kmer_ in tqdm(x, total=len(x)):
                        

                        try:

                            prediction += self.probabilities[cluster_name][kmer_]

                        except:

                            prediction += self.defaultRatio[cluster_name]

                    if(maxx == None or prediction > maxx):

                        maxx = prediction
                        y = cluster_name

                Y.append([maxx, y])

            return Y

        except Exception as e:

            raise ValueError('Pass as input list or numpy array of texts')
            return

    def predictProba(self, X):
        '''
        compute the prediction probabilities using the child instance
        
        Arguments:
            X {list or numpy array} -- the list of gene sequences
        
        Returns:
            [list] -- computes the prediction probabilities of the clusters in the following format
                        [ [cluster1_probability,cluster2_probability,...,clustern_probability] ... for x in X ]
        '''


        Y = []      # output clusters

        try:

            for i in tqdm(range(len(X))):
                
                y = []                                          # initialization
                x = breakIntoKmer(X[i], self.K).split(' ')      # break the sequence into k-mers
                for cluster_name in tqdm(self.classes, total=len(self.classes)):
                    
                    prediction = self.prior[cluster_name]

                    for kmer_ in tqdm(x, total=len(x)):
                        
                        try:

                            prediction += self.probabilities[cluster_name][kmer_]

                        except:

                            prediction += self.defaultRatio[cluster_name]

                    
                    y.append(prediction)

                Y.append(y)

            return Y

        except Exception as e:

            raise ValueError('Pass as input list or numpy array of texts')
            return


