"""
Read and write HTK feature files.
This module reads and writes the acoustic feature files used by HTK\

Usage: from speech_io.feature_io import *

OBS = readscp(r'\\vilfblgpu020\D\users\zhashi\BNFeat\scps\BNDNN_BN_1300hr.mfc.head.fakechunk')


Author:         zhashi@microsoft.com
Last update:    2015/12

Bug fixed: self.filesize --> self.nsamples in the HTK header

"""

from struct import unpack, pack
import numpy

__author__ = 'Shixiong Zhang'
__version__ = "1.0.1"

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0000100  # has energy
_N = 0000200  # absolute energy suppressed
_D = 0000400  # has delta coefficients
_A = 0001000  # has acceleration (delta-delta) coefficients
_C = 0002000  # is compressed
_Z = 0004000  # has zero mean static coefficients
_K = 0010000  # has CRC checksum
_O = 0020000  # has 0th cepstral coefficient
_V = 0040000  # has VQ data
_T = 0100000  # has third differential coefficients


def hopen(f, mode=None, vec_len=13):
    """Open an HTK format feature file for reading or writing.
    The mode parameter is 'rb' (reading) or 'wb' (writing)."""
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return HTKFeat_read(f)  # vec_len is ignored since it's in the file
    elif mode in ('w', 'wb'):
        return HTKFeat_write(f, vec_len)
    else:
        raise Exception, "mode must be 'r', 'rb', 'w', or 'wb'"


class HTKFeat_read(object):
    "Read HTK format feature files"

    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)

    def __iter__(self):
        self.fh.seek(12, 0)
        return self

    def open(self, filename):
        self.filename = filename
        self.fh = file(filename, "rb")
        self.readheader()

    def readheader(self):
        self.fh.seek(0, 0)
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = \
            unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'
            self.veclen = self.sampSize / 4
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)

    def next(self):
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getall(self):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)
        if self.parmKind & _K:  # Remove and ignore checksum
            if (len(data) / self.veclen == self.nSamples):
                print "MFC Header XXXX_K is not true! There is no additional checksum at the end of the file! Ingore the XXXX_K in the header, this will not affect anything."
            else:
                data = data[:-1]
        data = data.reshape(len(data) / self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

    def getchunk(self, startframe,
                 endframe):  # zhashi: support for chunkread   # .\chunk\archive.0[296,735]
        self.seek(startframe)
        data = numpy.fromfile(self.fh, self.dtype, (endframe - startframe + 1) * self.veclen)

        data = data.reshape(endframe - startframe + 1, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data


class HTKFeat_write(object):
    "Write HTK format feature files"

    def __init__(self, filename=None,
                 veclen=13, sampPeriod=100000,
                 paramKind=(MFCC | _O)):  # could be any kind above
        self.veclen = veclen
        self.sampPeriod = sampPeriod
        self.sampSize = veclen * 4
        self.paramKind = paramKind
        self.dtype = 'f'
        self.nsamples = 0
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open(filename)

    def __del__(self):
        self.close()

    def open(self, filename):
        self.filename = filename
        self.fh = file(filename, "wb")
        self.writeheader()

    def close(self):
        self.writeheader()

    def writeheader(self):
        self.fh.seek(0, 0)
        self.fh.write(pack(">IIHH", self.nsamples,
                           self.sampPeriod,
                           self.sampSize,
                           self.paramKind))

    def writevec(self, vec):        
        if len(vec) != self.veclen:
            raise Exception("Vector length must be %d" % self.veclen)
        if self.swap:
            numpy.array(vec, self.dtype).byteswap().tofile(self.fh)
        else:
            numpy.array(vec, self.dtype).tofile(self.fh)
        self.nsamples = self.nsamples + 1 #self.veclen

    def writeall(self, arr):
        # fix a spical case that arr only has 1 frame
        if arr.ndim==1:
            arr=[arr]            
        for row in arr:            
            self.writevec(row)


"""
==========================================================================================
"""

def write_one_mfc(filename, numpy_matrix):    
    if numpy_matrix.ndim==2:
        frame_num, fea_dim = numpy_matrix.shape           
    if numpy_matrix.ndim==1:
        fea_dim = numpy_matrix.shape[-1]        

    mfc = hopen(filename, 'wb', fea_dim)    
    mfc.writeall(numpy_matrix)

def read_one_mfc(filename, startframe=0, endframe=0):
    # """ read one mfcc file using htkmfc.py"""        
    mfc = hopen(filename)  # <htkmfc.HTKFeat_read object at 0x6ffffae1e90>  \\fbl\NAS\INVES\ruizhao\DNN\sequential\smdr3train\chunk\archive.0
    if (startframe == endframe == 0):
        obs = mfc.getall()  # or getchunk(startframe, endframe)  or getall()
    elif (endframe >= startframe):
        obs = mfc.getchunk(startframe, endframe)  # or getchunk(startframe, endframe)  or getall()
    else:
        print "startframe must be less than or equal to endframe"
    # frame_num, feat_dim = bnfea.shape
    return obs,mfc

def read_one_mfc_fast(mfc, startframe=0, endframe=0):
    # """ read one mfcc file using htkmfc.py"""            
    if (startframe == endframe == 0):
        obs = mfc.getall()  # or getchunk(startframe, endframe)  or getall()
    elif (endframe >= startframe):
        obs = mfc.getchunk(startframe, endframe)  # or getchunk(startframe, endframe)  or getall()
    else:
        print "startframe must be less than or equal to endframe"
    # frame_num, feat_dim = bnfea.shape
    return obs

def read_scp(filename):
    """Read scp files and the corresponding mfc files, Converted it to a dictionary"""


    o_utt = {}
    check_format = 0
    utt_dict = {}

    mfc_file_prev = 'XXXXXXYYY.mfc'
    mfcfp = None
    for line in file(filename):
        # Mobile_VS_Train_en-US_Live_06-2013/None/3cccdf83-49cb-4f92-bd96-ac1b63d9d553/7b79fb7e-ccc4-11e2-a953-001517a366d9.mfc=\\vilfblgpu020\D\users\zhashi\BNFeat\BNDNN_BN_1300hr\7b79fb7e-ccc4-11e2-a953-001517a366d9.mfc[0,489]

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
        
        mfc_file = line  # <------------- be careful, hard code for TIMIT        
        #print "processing", mfc_file

        # for os in comp.Win32_OperatingSystem():
        #    print float(os.FreePhysicalMemory)/1000, "KB of available memory"
        #    FreeMem=float(os.FreePhysicalMemory)
        #    MEM_perc=FreeMem/(TotalMemory+0.0)
        # print MEM_perc, 'percentage memeory usage. \n'
        # if MEM_perc<0.001:
        #    break
        # else:


        if (mfc_file == mfc_file_prev):
            obs = read_one_mfc_fast(mfcfp, startframe, endframe)            
        else:
            #mfcfp.close() # in python, file will be close automatically when run out of scope
            obs, mfcfp = read_one_mfc(mfc_file, startframe, endframe)  # zhashi: new archive file in the scp list
        mfc_file_prev =mfc_file

        if check_format:
            pure_name = set_file_name.split('.')[0]
        else:
            pure_name = mfc_file.split('\\')[-1].split('.')[0] #.split('\\')[-1][:-4]            
        # print "O_purename:", pure_name

        # Create and utterance dictionary to filter the MLF loading
        if pure_name not in utt_dict:
            utt_dict[pure_name] = 1
        o_utt[pure_name] = obs


    return o_utt, utt_dict

    # key = Mobile_SMD_en-US_Live_R3/44340/44340/45940
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
