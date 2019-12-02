import pickle
from tqdm import tqdm as tqdm
import numpy as np
from scipy.stats import skew, kurtosis
from ScalableNaiveBayes import *


def computeProbabilities(sequences, model_name, nChild):

    probabilities = []

    for i in range(len(sequences)):

        probabilities.append([])

    for child_id in tqdm(range(1,nChild+1)):

        child = pickle.load(open(os.path.join(model_name, str(child_id)+'.p'), 'rb'))

        probabilit_ = child.predictProba(sequences)        

        for i in range(len(sequences)):

            probabilities[i].extend(probabilit_[i])

            

    return probabilities


def featureExtraction(valid_probas, outlier_probas):

    X = []
    Y = []

    for valid_proba in tqdm(valid_probas):

        proba = np.array(valid_proba)

        proba = proba - np.min(proba)
        proba = proba / np.max(proba)

        mean = np.mean(proba)
        std = np.std(proba)
        summ = np.sum(proba)
        sk = skew(proba)
        kur = kurtosis(proba)

        x = [mean, std, summ, sk, kur]

        X.append(x)

        Y.append('Valid')

    
    for outlier_proba in tqdm(outlier_probas):

        proba = np.array(outlier_proba)

        proba = proba - np.min(proba)
        proba = proba / np.max(proba)

        mean = np.mean(proba)
        std = np.std(proba)
        summ = np.sum(proba)
        sk = skew(proba)
        kur = kurtosis(proba)

        x = [mean, std, summ, sk, kur]

        X.append(x)

        Y.append('Outlier')

    return (X, Y)


