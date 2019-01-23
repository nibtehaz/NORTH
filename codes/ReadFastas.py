import requests
import pickle
import os
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from FeatureExtraction import breakIntoKmer
import numpy as np 

def parseFiles():

	X = []

	genes = list(SeqIO.parse(fastaFile,'fasta'))



def readFastas(family,saveFile='3merFasta.p',k=3,nClasses=100):

	X = []
	Y = []

	files = next(os.walk('kegg/'+family+'/Fasta/'))[2]
	lenli = []

	for file in tqdm(files,total=len(files)):

		genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))
		lenli.append(len(genes))

	lenli.sort()

	print(lenli[0],lenli[-nClasses],lenli[-1])

	for file in tqdm(files,total=len(files)):

		genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))

		if(len(genes)<lenli[-nClasses]):

			continue

		np.random.shuffle(genes)

		for gene in genes:

			X.append(breakIntoKmer(str(gene.seq),k))
			Y.append(file[0:6])



	print(set(Y))

	print(len(set(Y)))

	pickle.dump((X,Y),open(saveFile,'wb'))


def rawReadFastas(family,saveFile='3merFasta.p',nClasses=100,minCnt=None,K=3):

	X = []
	Y = []

	vocabulary = {}

	files = next(os.walk('kegg/'+family+'/Fasta/'))[2]
	lenli = []

	if(minCnt==None):

		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))
			lenli.append(len(genes))

		lenli.sort()

		print(lenli[0],lenli[-nClasses],lenli[-1])

		lenli[-nClasses] = max(lenli[-nClasses],20)

		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))

			if(len(genes)<lenli[-nClasses]):

				continue

			np.random.shuffle(genes)

			for gene in genes:

				X.append(str(gene.seq))
				Y.append(file[0:6])

				kmers = breakIntoKmer(str(gene.seq),K).split(' ')

				for kmer in tqdm(kmers,total=len(kmers)):

						vocabulary[kmer] = True 

		vSize = len(vocabulary)


	else:

		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))

			if(len(genes)<minCnt):

				continue

			for gene in genes:

				kmers = breakIntoKmer(str(gene.seq),K).split(' ')

				for kmer in tqdm(kmers,total=len(kmers)):

						vocabulary[kmer] = True 

		vSize = len(vocabulary)

		vocabulary = 'garbage'
		
		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse('kegg/'+family+'/Fasta/'+file,'fasta'))

			if(len(genes)<minCnt):

				continue

			np.random.shuffle(genes)

			for gene in genes:

				X.append(str(gene.seq))
				Y.append(file[0:6])


	print(set(Y))

	print(len(set(Y)))

	print(vSize)

	pickle.dump((X,Y,vSize),open(saveFile,'wb'))


if __name__=='__main__':


	
	#readFastas('eukaryotes','3mer.p',k=3,nClasses=20)

	rawReadFastas('completed/all','3merAll6.p',nClasses=200,minCnt=None,K=5)