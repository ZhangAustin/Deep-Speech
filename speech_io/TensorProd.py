"""
Input 1: Feature files in HTK format
Input 2: Posterior files in HTK format

Output: the tensor product of Input 1 and 2 in HTK format


Usage: from speech_io.feature_io import *

TensorProd(SCP_Fea, SCP_post, SCP_output_Tensor)

Author:		 zhashi@microsoft.com
Last update:	2017/05

Dependency: https://github.com/ZhangAustin/Deep-Speech

"""

import sys
import numpy as np
from feature_io import *	# usage: from speech_io import feature_io	==> feature_io.read_one_mfc()
							# usage: from speech_io.feature_io import *  ==> read_one_mfc()


def Tensor_VectorSeq_Avg(seq_a, seq_b, Avg_orNot=0): # produce the feature sequence [post_t * o_t] or avg utterance-level feature sum_t [post_t * o_t]
	assert(seq_a.shape[0] == seq_b.shape[0])
	nframe=seq_a.shape[0]
	tensor_dim = seq_a.shape[1] * seq_b.shape[1]
	if Avg_orNot==0:
		output_matrix = np.empty((nframe,tensor_dim), dtype='float32')
	else:
		output_avgVec = np.zeros(tensor_dim)

	for t in xrange(0, nframe):
		if Avg_orNot==0:
			output_matrix[t] = np.kron(seq_a[t],seq_b[t])
		else:
			output_avgVec = output_avgVec + np.kron(seq_a[t],seq_b[t])

	if Avg_orNot==0:
		return output_matrix
	else:
		return output_avgVec/nframe		#<------------- average by T


def Tensor_feature_generator(seq_a, seq_b, softT_or_hardT=1):
	# assume seq_a is the post_t
	# produce the final utterance-level feature sum_t [post_t * o_t] / sum_t(post_t), the sum of post_t can be viewed as soft duration.
	assert(seq_a.shape[0] == seq_b.shape[0])
	nframe=seq_a.shape[0]	
	tensor_dim = seq_a.shape[1] * seq_b.shape[1]
	output_Vec = np.zeros(tensor_dim)
	

	for t in xrange(0, nframe):
		output_Vec = output_Vec + np.kron(seq_a[t],seq_b[t])
	
	if softT_or_hardT:
		soft_nframe=np.zeros(tensor_dim)
		for t in xrange(0, nframe):
			soft_nframe=soft_nframe+np.kron(seq_a[t],np.ones(seq_b.shape[1]))
		return output_Vec/soft_nframe 
	else:
		return output_Vec/nframe 
	
"""
To do:
1) generative kernel feature generator
2) derivative kernel feature generator

details: https://ai2-s2-pdfs.s3.amazonaws.com/ad1f/6b77c4cad77f7468758df088b4806f742acf.pdf
"""

def TensorProduct_scp(SCP_Fea, SCP_post, Output_SCP, softT_or_hardT=1):

	"""Read scp files and the corresponding mfc files

	example of SCP:
	# Mobile_VS_Train_en-US_Live_06-2013/None/3cccdf83-49cb-4f92-bd96-ac1b63d9d553/7b79fb7e-ccc4-11e2-a953-001517a366d9.mfc=\\vilfblgpu020\D\users\zhashi\BNFeat\BNDNN_BN_1300hr\7b79fb7e-ccc4-11e2-a953-001517a366d9.mfc[0,489]
	# Mobile_VS_Train_en-US_Live_06-2013/None/3cccdf83-49cb-4f92-bd96-ac1b63d9d553/69934j43-erc3-45e6-b8u9-5956937736c8.mfc=\\vilfblgpu020\D\users\zhashi\BNFeat\BNDNN_BN_1300hr\7b79fb7e-ccc4-11e2-a953-001517a366d8.mfc[490,689]
	"""	
	o_utt, utt_dict_feat =read_scp(SCP_Fea)
	p_utt, utt_dict_post =read_scp(SCP_post)

	# key = Mobile_SMD_en-US_Live_R3/44340/44340/45940
	# one row one frame
	# o_utt[key]=[[  3.84167051e+00   2.94051337e+00   4.03393555e+00 ...,   3.89492586e-02
				 #   4.45630476e-02  -4.89231683e-02]
				 #[  4.10013390e+00   2.44905114e+00   2.55832839e+00 ...,   4.76126708e-02
				 #   3.57627682e-02  -9.29152817e-02]
				 #[  3.67069077e+00   3.31796169e+00   4.15470505e+00 ...,   3.51629220e-02
				 #   3.53568085e-02  -9.80582908e-02]
				 #...,
				 #[  4.43760490e+00   3.17837548e+00   3.14262390e+00 ...,  -2.42370795e-02
				 #  -2.02618632e-02   4.84732911e-03]
				 #[  4.38527155e+00   2.22882104e+00   3.36550379e+00 ...,  -2.64272355e-02
				 #  -2.57759001e-02  -2.50774412e-03]
				 #[  4.76648808e+00   2.39261246e+00   3.40049219e+00 ...,  -1.77562032e-02
				 #  -2.36367565e-02  -4.42066835e-03]]

	check_format = 0
	for line in file(Output_SCP):
		
		if line.find('#'): line = line[:line.find('#')]  # Get rid of comments.
		line = line.split()[0]  # If the line is empty, who cares?
		if not line: continue

		if '=' in line:
			set_file_name, line = line.split('=')
			check_format = 1
		mfc_file = line  # <------------- 
		if check_format:
			pure_name = set_file_name.split('.')[0]
		else:
			pure_name = mfc_file.split('\\')[-1].split('.')[0]

		if pure_name not in utt_dict_feat:
			print "output file name can not be found in the feature scp"
			break

		Tensor_uttlevel_feature=Tensor_feature_generator(o_utt[pure_name],p_utt[pure_name], softT_or_hardT)		
		write_one_mfc(mfc_file, Tensor_uttlevel_feature)



SCP_Fea=sys.argv[1]
SCP_post=sys.argv[2]
Output_SCP=sys.argv[3]
if len(sys.argv)>4:
	softT_or_hardT=sys.argv[4]
TensorProduct_scp(SCP_Fea, SCP_post, Output_SCP, softT_or_hardT=1)

