"""
Input : archive scp (frame based MFCCs)
Output: archive scp (utterance based MFCCs)


Usage: python frame2utt_feature input.scp output.scp

Author:		 zhashi@microsoft.com
Last update:	2017/05

Dependency: https://github.com/ZhangAustin/Deep-Speech

"""

import sys
import numpy as np
from feature_io import *	# usage: from speech_io import feature_io	==> feature_io.read_one_mfc()
							# usage: from speech_io.feature_io import *  ==> read_one_mfc()


def avg_length(archive_scp):	
	total_length=0
	num_utt=0
	max_len=0
	min_len=999
	for line in file(archive_scp):
        # 39DDCBFDBE134C83A47A0DAF1EDD480F_0.mfc=16kmfcc.kwhead.archive/archive.1[0,52]
		startframe = 0
		endframe = 0
		if line.find('#'): line = line[:line.find('#')]  # Get rid of comments.
		line = line.split()[0]  # If the line is empty, who cares?        
		if not line: continue
		if '=' in line:
		    set_file_name, line = line.split('=')            
		    check_format = 1
		if '[' in line:
			line, rest = line.split('[')
			startframe = int(rest.split(',')[0])
			endframe = int(rest.split(',')[1].split(']')[0])
			total_length=total_length+endframe-startframe+1
			num_utt = num_utt +1
			if (endframe-startframe+1) > 198:
				print "file ", set_file_name
			if (endframe-startframe+1) > max_len:
				max_len = endframe-startframe+1				
			if (endframe-startframe+1) < min_len:
				min_len = endframe-startframe+1
	avg_length= total_length / num_utt
	return avg_length, max_len, min_len



def frame2utt_hardcut(SCP_Fea, Output_Big_MFC, Uttlen=80):

	"""Read scp files and the corresponding mfc files

	example of SCP:
	# 39DDCBFDBE134C83A47A0DAF1EDD480F_0.mfc=16kmfcc.kwhead.archive/archive.1[0,52]
	# 6F72B1A97AD24607B680A315C7D19DFE_1.mfc=16kmfcc.kwhead.archive/archive.1[53,137]
	"""	
	#o_utt, utt_dict_feat =read_scp(SCP_Fea)
	line_counter = 0	
	for line in file(SCP_Fea):
		startframe = 0
		endframe = 0
		if line.find('#'): line = line[:line.find('#')]  # Just in case, someone like to make comments in the scp... trust me that happened before.
		line = line.split()[0]  # If the line is empty, who cares? 
		if not line: continue

		if '=' in line:
			set_file_name, line = line.split('=')
			check_format = 1			
		if '[' in line:
			line, rest = line.split('[')
			startframe = int(rest.split(',')[0])
			endframe = int(rest.split(',')[1].split(']')[0])        
			line_counter = line_counter + 1

		mfc_file = line  
		obs = read_one_mfc(mfc_file, startframe, endframe) # <------------- 			
		obs = np.pad(obs, [(Uttlen, 0), (0, 0)], 'constant')		
		tmp_vec = obs[-Uttlen:,:].flatten()
		if (line_counter==1):
			uttlevel_fea = tmp_vec		
		uttlevel_fea = np.vstack((uttlevel_fea,tmp_vec))
		


	write_one_mfc(Output_Big_MFC, uttlevel_fea)
        
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



SCP_Fea=sys.argv[1]
Output_Big_MFC=sys.argv[2]
if len(sys.argv)>3:
	Uttlen=int(sys.argv[3])	
	frame2utt_hardcut(SCP_Fea, Output_Big_MFC, Uttlen)
else:
	frame2utt_hardcut(SCP_Fea, Output_Big_MFC)


