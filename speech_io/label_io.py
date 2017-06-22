"""
Read and write HTK MLF files.
This module reads and writes the label files in HTK format

Usage: from speech_io.label_io import *

s_utt, si_utt = read_mlffile(MLFFile, utt_dict, dict_state)

where, utt_dict is the utterance list in SCP file
where, dict_state is the senone list

Author:         zhashi@microsoft.com
Last update:    2015/12
"""

from __future__ import print_function


import numpy as np


def read_prior(filename):
    dict_prior={}
    for line in file(filename):
        if not line: continue

        ind, prior = line.split()
        ind = int(ind)
        prior = float(prior)
        dict_prior[ind]=np.log(prior)           #ind=[0,1,...182]

    return dict_prior

def read_mlffile(filename, utt_dict, dict_state):
    """Read mlf files and the corresponding state labels"""
    # mlf file
    # "//svmtdata/DNNData/V/kakalgao/timit/data/train/dr1/fcjf0/si1027.lab"
    # 0 200000 h#_s2 -210.2248 h# -624.4926 h#
    # 200000 400000 h#_s3 -210.6809
    # 400000 600000 h#_s4 -203.5869
    # 600000 1600000 q_s2 -1244.831 h#-q+b -1467.823 q
    # 1600000 1700000 q_s3 -117.3826
    # 1700000 1800000 q_s4 -105.61
    # ...
    # .
    MFCC_TIMESTEP = 10    # the frame shift in terms of ms

    # Convert the list to dictionary
    #s_utt = {}
    si_utt = {}
    for line in open(filename):
        line = line.rstrip()    #.rstrip('\n')
        if len(line) < 1:
            continue
        #one MFC
        if line[0] == '"': # a new utterance
            filename_s = line.strip('"')
            #pure_filename_s = filename_s.split('/')[-2] + '_' + filename_s.split('/')[-1][:-4]    #<------------- be careful, hard code for TIMIT
            pure_filename_s = filename_s[:-4]
            #print "purename:", pure_filename_s
            #s = ()
            si = ()

        #one MLF
        elif ( (pure_filename_s in utt_dict) and line[0].isdigit() ):
            start, end, state = line.split()[:3]

            #if ']' in state:
            #    state=state.rstrip(']').replace('[','_s')

            # start and end frame index
            if (int(end)>=100000):
                start = int((int(start)+1)/(MFCC_TIMESTEP * 10000))
                end = int((int(end)+1)/(MFCC_TIMESTEP * 10000))
            else:
                start = int(start)
                end = int(end)
                #print start, end, state

            for t in range(start, end):
                #s = s + (state,)   # append the state label for each frame
                si = si + (dict_state[state],)

        if line[0] == '.': # utterance END
            #print("reading MLF: ", filename_s)
            if pure_filename_s in utt_dict:
                #s_utt[pure_filename_s] = s
                si_utt[pure_filename_s] = si
                #print('Adding %s file to record'%pure_filename_s)

    return si_utt  #s_utt,
    # len(s_utt)
    # s_utt['fcjf0_si1657'] = ('h#_s2', 'h#_s3', 'h#_s4', 'q_s2', 'q_s2', 'q_s2', ......, 'th_s4', 'th_s4', 'th_s4', 'h#_s2', 'h#_s2', 'h#_s2', 'h#_s3', 'h#_s3', 'h#_s3', 'h#_s3', 'h#_s3', 'h#_s3', 'h#_s4')
    # si_utt['fcjf0_si1657']= (1,2,3,46,46,46,...)





def read_statelist(filename):

    dict_state={}
    ind = 1
    for line in open(filename):

        state = line.rstrip()
        #ind, state = line.split()
        #state=state.rstrip(']').replace('[','_s')
        dict_state[state]=ind
        ind = ind + 1

    print('\n', len(dict_state), 'states mapped to [1,2,3,...%d], the state and index mapping are: ' % (len(dict_state)) )
    print(dict_state,'\n')
    return dict_state
    # {'m_s4': 114, 'ch_s2': 34, 'ch_s3': 35, 'ch_s4': 36, 'ux_s4': 168, 'm_s3': 113, 'bcl_s4': 33, 's_s4': 147, 's_s3': 146, 's_s2': 145, 'oy_s3': 128, 'oy_s2': 127, 'oy_s4': 129, 'uh_s2': 160, 'uh_s3': 161, 'bcl_s3': 32, 'uh_s4': 162, 'pau_s2': 133, 'pau_s3': 134, 'ah_s4': 9, 'bcl_s2': 31, 'ah_s2': 7, 'ah_s3': 8, 'pau_s4': 135, 'sh_s3': 149, 'b_s4': 30, 'dx_s2': 46, 'ax_s4': 18, 'ax_s2': 16, 'ax_s3': 17, 'b_s2': 28, 'b_s3': 29, 'k_s4': 105, 'k_s3': 104, 'k_s2': 103, 'gcl_s2': 79, 'gcl_s3': 80, 'gcl_s4': 81, 'z_s3': 179, 'ix_s4': 96, 'ix_s2': 94, 'ix_s3': 95, 'sh_s2': 148, 'axh_s3': 20, 'axh_s2': 19, 'y_s3': 176, 'y_s2': 175, 'p_s4': 132, 'axh_s4': 21, 'en_s2': 58, 'en_s3': 59, 'en_s4': 60, 'p_s2': 130, 'l_s2': 109, 'l_s3': 110, 'l_s4': 111, 't_s2': 151, 'aa_s4': 3, 'aa_s3': 2, 'aa_s2': 1, 'w_s3': 173, 'w_s2': 172, 'w_s4': 174, 'q_s3': 140, 'q_s2': 139, 'sh_s4': 150, 'q_s4': 141, 'h#_s4': 84, 'h#_s2': 82, 'h#_s3': 83, 't_s4': 153, 'r_s4': 144, 'zh_s3': 182, 'zh_s2': 181, 'zh_s4': 183, 'r_s2': 142, 'r_s3': 143, 'g_s3': 77, 'g_s2': 76, 'ow_s3': 125, 'g_s4': 78, 'ow_s4': 126, 'z_s4': 180, 't_s3': 152, 'y_s4': 177, 'ao_s3': 11, 'ao_s2': 10, 'ao_s4': 12, 'aw_s3': 14, 'aw_s2': 13, 'pcl_s4': 138, 'aw_s4': 15, 'nx_s4': 123, 'axr_s4': 24, 'axr_s3': 23, 'axr_s2': 22, 'ow_s2': 124, 'epi_s2': 64, 'epi_s3': 65, 'epi_s4': 66, 'uw_s4': 165, 'uw_s3': 164, 'uw_s2': 163, 'ay_s4': 27, 'eh_s2': 49, 'eh_s3': 50, 'hv_s3': 89, 'hv_s2': 88, 'hv_s4': 90, 'ay_s3': 26, 'ay_s2': 25, 'v_s2': 169, 'v_s3': 170, 'v_s4': 171, 'dcl_s3': 41, 'dcl_s2': 40, 'eh_s4': 51, 'ng_s2': 118, 'dcl_s4': 42, 'eng_s2': 61, 'eng_s3': 62, 'eng_s4': 63, 'dx_s3': 47, 'nx_s3': 122, 'jh_s3': 101, 'jh_s2': 100, 'jh_s4': 102, 'el_s4': 54, 'el_s2': 52, 'el_s3': 53, 'f_s2': 73, 'f_s3': 74, 'f_s4': 75, 'p_s3': 131, 'tcl_s4': 156, 'm_s2': 112, 'dx_s4': 48, 'ng_s4': 120, 'd_s2': 37, 'd_s3': 38, 'hh_s4': 87, 'hh_s3': 86, 'hh_s2': 85, 'd_s4': 39, 'nx_s2': 121, 'tcl_s3': 155, 'tcl_s2': 154, 'n_s2': 115, 'n_s3': 116, 'n_s4': 117, 'iy_s4': 99, 'em_s4': 57, 'ey_s3': 71, 'ey_s2': 70, 'z_s2': 178, 'ey_s4': 72, 'em_s3': 56, 'em_s2': 55, 'kcl_s4': 108, 'kcl_s2': 106, 'kcl_s3': 107, 'dh_s3': 44, 'dh_s2': 43, 'ng_s3': 119, 'pcl_s3': 137, 'pcl_s2': 136, 'dh_s4': 45, 'iy_s3': 98, 'iy_s2': 97, 'ae_s3': 5, 'ae_s2': 4, 'ae_s4': 6, 'th_s3': 158, 'th_s2': 157, 'th_s4': 159, 'ux_s2': 166, 'ux_s3': 167, 'er_s4': 69, 'ih_s4': 93, 'ih_s2': 91, 'ih_s3': 92, 'er_s2': 67, 'er_s3': 68}
    # dict_state['m_s4'] = 114


def init_TRANSP(dim):
    TRANSP = np.zeros((dim,dim))
    for i in range(dim):
        if (i==0):
            TRANSP[0,1] = 1.0
        elif (i<dim-1):
            TRANSP[i,i] = 0.5
            TRANSP[i,i+1] = 0.5
    return TRANSP



def read_MMF(filename):

    dict_MMF={}
    inside = 0

    f_MMF = open(filename)
    while True:
        line = f_MMF.readline()


        if line.startswith ('~h') and inside == 0:   # ~h "b"
            if len (line.split()) > 1:
                model_name = line.split()[1].strip("\"")           # model_name='b'
                #print model_name

        else:
            if line.startswith ('<BEGINHMM>'):      # <BEGINHMM>
                inside += 1
            if line.startswith ('<NUMSTATES>'):     # <NUMSTATES>
                NUMSTATES = line.split()[1]
            if line.startswith ('<STATE>'):         # <STATE>
                state_234 = int(line.split()[1])
            if line.startswith ('<MEAN>'):          # <MEAN>    Note that for MMF.cdbn file the <MEAN> is actually the index
                line = f_MMF.readline()
                #state_index = int(line)
            if line.startswith ('<TRANSP>'):        # <TRANSP>
                TRANSP_dim = int(line.split()[1])
                TRANSP = init_TRANSP(TRANSP_dim)
                for j in range(TRANSP_dim):
                    line = f_MMF.readline()
                    TRANSP[j,:] = np.array ([float (entry) for entry in line.split()])
            if line.startswith ('<ENDHMM>'):        # <ENDHMM>
                inside -= 1
                # finish one phone HMM
                dict_MMF[model_name]=TRANSP

        if not line:
            break
    f_MMF.close()
    return dict_MMF
    #'b': array([[ 0.       ,  1.       ,  0.       ,  0.       ,  0.       ],
    #            [ 0.       ,  0.4667291,  0.5332709,  0.       ,  0.       ],
    #            [ 0.       ,  0.       ,  0.2659557,  0.7340443,  0.       ],
    #            [ 0.       ,  0.       ,  0.       ,  0.3781517,  0.6218483],
    #            [ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ]])


