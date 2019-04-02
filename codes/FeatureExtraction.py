def breakIntoKmer(inp,k):
	'''
	Breaks a protein sequence into k-mer sequence
	
	Arguments:
		inp {str} -- The input sequence
		k {int} -- value of k as in k-mer
	
	Returns:
		str -- The k-mer sequence
	'''


	out = ''

	for i in range(k-1,len(inp)):

		for j in range(1,k+1):

			out += inp[i-k+j]

		out += ' '

	return out[:-1]



