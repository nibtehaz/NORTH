import pickle
import os
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from FeatureExtraction import breakIntoKmer
import numpy as np 
from time import sleep


def readFastas(data_path, save_file_name='3merFasta.p', n_clusters=100, min_cnt=None, K=3):
	'''
	Reads the fasta files from the directory
	
	If we provide value to min_cnt variable then all the clusters having a minimum of
	min_cnt number of genes will be considered. 
	*If this is given, the n_clusters value will be ignored*

	Otherwise, if we provide value to n_clusters variable then only the biggest
	n_clusters clusters will be considered

	Arguments:
		data_path {str} -- path to the directory containing fasta files
	
	Keyword Arguments:
		save_file_name {str} -- name of the pickle file that contains all the gene sequence (default: {'3merFasta.p'})
		n_clusters {int} -- number of ortholog clusters (default: {100})
		min_cnt {int} -- minimum number of genes in a cluster to consider (default: {None})							
		K {int} -- value of k (default: {3})
	'''


	X = []			# list to store gene sequences
	Y = []			# list to store ortholog cluster ids of individual genes

	vocabulary = {}			# dictionary to contain vocabulary

	files = next(os.walk(data_path))[2]
	lenli = []				# list containing size of ortholog clusters

	if(min_cnt==None):				# Considering the biggest n_clusters clusters

		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse(os.path.join(data_path,file),'fasta'))
			lenli.append(len(genes))
			

		lenli.sort()

		print('Smallest Cluster : {} genes'.format(lenli[0]))
		print('{}th Cluster : {} genes'.format(n_clusters, lenli[-n_clusters]))
		print('Biggest Cluster : {} genes'.format(lenli[-1]))

		lenli[-n_clusters] = max(lenli[-n_clusters],20)		# If the clusters are too small
															# this avoids empty clusters


		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse(os.path.join(data_path,file),'fasta'))

			if(len(genes)<lenli[-n_clusters]):		# ignoring smaller clusters

				continue

			np.random.shuffle(genes)				# randomly shuffling the genes 

			for gene in genes:					

				X.append(str(gene.seq))				# adding the sequence
				Y.append(file[:-6])					# adding the cluster id
													# as the last 6 characters are '.fasta'
																
				kmers = breakIntoKmer(str(gene.seq),K).split(' ')		

				for kmer in (kmers): 		
				#debug line for kmer in tqdm(kmers,total=len(kmers)): 		

						vocabulary[kmer] = True 				# computing vocabulary
						
		vSize = len(vocabulary)


	else:							# Considering clusters with at least min_cnt genes

		for file in tqdm(files,total=len(files)):

			genes = list(SeqIO.parse(os.path.join(data_path,file),'fasta'))

			if(len(genes)<min_cnt):				# ignoring smaller clusters

				continue

			for gene in genes:

				kmers = breakIntoKmer(str(gene.seq),K).split(' ')

				for kmer in tqdm(kmers,total=len(kmers)):

						vocabulary[kmer] = True 				# computing vocabulary
						

		vSize = len(vocabulary)

		vocabulary = 'garbage'		# literal garbage collection :p 
		
		for file in tqdm(files,total=len(files)):
			

			genes = list(SeqIO.parse(os.path.join(data_path,file),'fasta'))

			if(len(genes)<min_cnt):		# ignoring smaller clusters

				continue

			np.random.shuffle(genes)			# randomly shuffling the genes 

			for gene in genes:

				X.append(str(gene.seq))			# adding the sequence
				Y.append(file[:-6])				# adding the cluster id
												# as the last 6 characters are '.fasta'

	
	print('Clusters : {}'.format(set(Y)))			# prints the clusters

	print('Total clusters : {}'.format(len(set(Y))))		# prints the total number of clusters

	print('Vocabulary Size : {}'.format(vSize))

	pickle.dump((X,Y,vSize),open(save_file_name,'wb'))			# stroing the data to be used in training later


if __name__=='__main__':

	readFastas('completed/all','3merAll6.p',n_clusters=200,min_cnt=None,K=5)