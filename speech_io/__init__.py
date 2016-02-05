
"""
Package name:   Speech_IO
Usage       :   from speech_io import *
Description :   read htk features, mlf labels, keras(HDF5)/pickle models
                write log files, HDF5/pickle models

Author:         zhashi@microsoft.com
Last update:    2015/12


-----------------------------------------------------
speech_io/ __init__.py
    feature_io.py
    label_io.py
    model_io.py
    Module2.py

    sub_package/ __init__.py
       sub_package_module1.py
       sub_package_module2.py
-----------------------------------------------------
"""

from feature_io import *    # usage: from speech_io import feature_io    ==> feature_io.read_one_mfc()
                            # usage: from speech_io.feature_io import *  ==> read_one_mfc()
from label_io import *      # usage: from speech_io import label_io
from model_io import *      # usage: from speech_io import model_io



__all__ = ['feature_io', 'label_io', 'model_io']      # usage: from speech_io import *   ===> feature_io.read_one_mfc()



######################################################################
# THE FOLLOWING IS AN EXAMPLE TO USE THIS PACKAGE
######################################################################

def read_examples(configfile):       # scp file: //svmtdata/DNNData/V/kakalgao/timit/data/train/dr1/fcjf0/si1027.mfc


    """ Get the MLF and SCP filenames """
    # Open the SCP and MLF from config file .
    for linefile in open(configfile):

        if linefile.find('#'): linefile = linefile[:linefile.find('#')] # Get rid of comments.
        linefile = linefile.split()[0]   # If the line is empty, who cares?
        if not linefile: continue

        if linefile[:3] == 'SCP':  SCPfile=linefile.split('=')[1]
        if linefile[:3] == 'MLF':  MLFfile=linefile.split('=')[1]
        if linefile[:3] == 'STA':  STATEfile=linefile.split('=')[1]
        if linefile[:3] == 'MMF':  MMFfile=linefile.split('=')[1]
        if linefile[:3] == 'NLM':  Ngramfile=linefile.split('=')[1]
        if linefile[:3] == 'PRI':  Priorfile=linefile.split('=')[1]

    """Read scp files and the corresponding mfc files"""
    # scp file:
    # //svmtdata/DNNData/V/kakalgao/timit/data/train/dr1/fcjf0/si1027.mfc
    o_utt, utt_dict =read_scpfile(SCPfile)


    """Read mlf files and the corresponding state labels"""
    s_utt=read_mlffile(MLFfile, utt_dict)


    """Read statelist """
    dict_state = {}
    #dict_state=read_statelist(STATEfile)

    """Read cdbn """
    #dict_MMF=read_MMF(MMFfile)
    dict_MMF = {}

    """Read ngram """
    #dict_Ngram=read_NgramLat(Ngramfile)


    """Read state prior """
    dict_prior = {}
    #dict_prior=read_prior(Priorfile)

    """ couple O_{1:T} and s_{1:T} for utt=1,...,R """
    examples = []
                           # check if miss files
    for fname,target in s_utt.items():
        r = 1
        if fname in o_utt:
            # Get the size of the array
            frame_num, frame_dim = o_utt[fname].shape
            if (frame_num == len(target)):         # Check if the features count and target id's match
                examples.append( (o_utt[fname],target) )    #<-------------------- all the examples[r]=(x_{1:T},s_{1:T})
            else:
                print("File:", o_utt[fname], " ERROR: Time frame of x_t=", frame_num, "and s_t=", len(target),"not match! \n"),
                sys.exit()
        else:
            print("%s File Missing in the obs/or features \n"%fname),



    print(len(examples),'examples read', ' feature dim is', frame_dim)
    return examples, dict_state, dict_prior

