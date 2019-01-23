import numpy as np
from classes import GeneFasta
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt 
import re

FOLDER_CNT = 407

def createKmerData(k=3,saveFile='3mer.p'):

	X = []
	Y = []

	maxx=0

	for i in tqdm(os.listdir('data/'),total=len(os.listdir('data/'))):
		
		if(len(os.listdir('data/'+i))<FOLDER_CNT):

			continue	

		print(len(os.listdir('data/'+i)))

		for j in os.listdir('data/'+i):

			#print('data/'+i+'/'+j)

			data = pickle.load(open('data/'+i+'/'+j,'rb'))

			maxx=max(maxx,len(data.AASeq))

			X.append(breakIntoKmer(data.AASeq,k))

			Y.append(i)


	print(maxx)

	print(set(Y))

	print(len(set(Y)))

	pickle.dump((X,Y),open(saveFile,'wb'))

def dataDistribution():

	li = []
	
	for i in tqdm(os.listdir('data/'),total=len(os.listdir('data/'))):
		
		li.append(len(os.listdir('data/'+i)))


	li.sort()
#	plt.hist(li)

#	plt.show()

	print(li[0])
	print(li[-1])
	print(li[-200])


def loadKmerData(fileNeam='3mer.p'):

	(X,Y) = pickle.load(open(fileNeam,'rb'))

	return (X,Y)

def breakIntoKmer(inp,k):

	out = ''

	for i in range(k-1,len(inp)):

		for j in range(1,k+1):

			out += inp[i-k+j]

		out += ' '

	return out[:-1]

def vectorize(X,Y):

	vectorizer = CountVectorizer()

	X = vectorizer.fit_transform(X)

	return (X,Y)



if __name__=='__main__':

	k = 3 

	#dataDistribution()

	#createKmerData(k,saveFile='3mer200good.p')
	li=set()




	for i in tqdm(os.listdir('data/'),total=len(os.listdir('data/'))):
	
		if(len(os.listdir('data/'+i))<FOLDER_CNT):

			continue	

		print(len(os.listdir('data/'+i)))

		for j in os.listdir('data/'+i):

			#print('data/'+i+'/'+j)

			li.add(re.split('1 2 3 4 5 6 7 8 9 0',j)[0])

	print(li)
	print(len(li))







