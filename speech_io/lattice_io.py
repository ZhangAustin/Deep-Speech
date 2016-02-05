# Read and write HTK lattices
# Do Forward-Backward Search

# Shi-Xiong (Austin) Zhang @ microsoft

import numpy as np
from collections import defaultdict, Counter
import operator
import time
#from LogProb import LogProb
#from JointProbability import JointProbability
import math

LOGZERO = -100000
MFCC_TIMESTEP = 10

# Log-probability class
negInf = float ('-inf')
class LogProb:
    def __init__ (self, scalar = None, exponent = None):
        if scalar != None:
            if scalar == 0:
                self.exponent = negInf
            else:
                self.exponent = math.log (float (scalar))
        else:
            self.exponent = exponent
    
    def getExponent (self):
        return self.exponent
    
    def getScalar (self):
        if self.exponent == negInf:
            return 0.
        else:
            return math.exp (self.exponent)
    
    def __float__ (self):
        return self.getScalar()
    
    def __str__ (self):
        return '%f' % (self.getScalar())
    
    def __repr__ (self):
        return '<LogProb exponent=%f, scalar=%f>' % (self.exponent, self.getScalar())
    
    def __add__ (self, other):
        if self.exponent == negInf:
            return other
        if other.exponent == negInf:
            return self
        # log (a + b) from log(a) and log(b), assuming a is largest:
        # log(a) + log (1 + exp (log(b) - log(a)))
        if self > other:
            loga = self.exponent
            logb = other.exponent
        else:
            loga = other.exponent
            logb = self.exponent
        return LogProb (exponent = loga + math.log (1 + math.exp (logb - loga)))
    
    def __mul__ (self, other):
        return LogProb (exponent = self.exponent + other.exponent)
    
    def __lt__ (self, other):
        return self.exponent < other.exponent
    def __le__ (self, other):
        return self.exponent <= other.exponent
    def __eq__ (self, other):
        return self.exponent == other.exponent
    def __ne__ (self, other):
        return self.exponent != other.exponent
    def __gt__ (self, other):
        return self.exponent > other.exponent
    def __ge__ (self, other):
        return self.exponent >= other.exponent
    
    def __hash__ (self):
        return hash (self.exponent)

class JointProbability:
    def __init__ (self, values):
        self.values = values
    
    def __str__ (self):
        return '%s' % self.values
    
    def __add__ (self, other):
        assert (len (self.values) == len (other.values))
        newValues = {}
        for (name, value) in self.values.iteritems():
            newValues [name] = value + other.values [name]
        return JointProbability (newValues)

    def __mul__ (self, other):
        assert (len (self.values) == len (other.values))
        newValues = {}
        for (name, value) in self.values.iteritems():
            newValues [name] = value * other.values [name]
        return JointProbability (newValues)

    def combined (self):
        return reduce (lambda x, y: x * y,
            [value for (name, value) in self.values.iteritems()])
    
    def __lt__ (self, other):
        return self.combined() < other.combined()
    def __le__ (self, other):
        return self.combined() <= other.combined()
    def __eq__ (self, other):
        return self.combined() == other.combined()
    def __ne__ (self, other):
        return self.combined() != other.combined()
    def __gt__ (self, other):
        return self.combined() > other.combined()
    def __ge__ (self, other):
        return self.combined() >= other.combined()


def extractValues (s):             # J=86    S=41   E=44   l=-76.05   d=:sil-j^I+M6w^F;1,0.04,-321.68:j^M-M6w^F;6+b^I,0.10,-593.76:sp,0.00,-0.05:
    s = s.split()
    values = {}
    for valueText in s:
        name, value = tuple (valueText.split ('=', 1))
        assert (not name in values)
        values [name] = value
    return values                   # {'J': '11920'}

def readProperty (s, name):         # name="J"/"t"/"W"
    assert (s.startswith (name + '='))
    return s [1 + len (name) :]     # "-76.05"

def readInt (s, name):
    return int (readProperty (s, name))

def readFloat (s, name):
    return float (readProperty (s, name))

class Node:
    def __init__ (self, line):
        if (len(line.split())==3):
            (time, word, rest) = tuple (line.split (None, 2))
        if (len(line.split())==2):
            (time, word) = tuple (line.split ())
            rest=''
        self.time = readFloat (time, 't')
        self.frame = int((self.time + 0.00001)*100)                  # convert to frame <----------- hard code
        self.word = readProperty (word, 'W')
        #if (self.word == '!EXIT' or self.word == '!ENTER'):          #<----------------------------- hard code, utterance start and end with sil but
        #    self.word = 'h#'
            
        self.description = rest.strip()
        self.arcs = []
        self.bestpathscore = LOGZERO
        self.bestprevarc = None

    
    def __str__ (self):
        return 'Node (%f %s %s)' % (self.time, self.word, self.description)
    
    def write (self, f):
        f.write ('t=%.2f\tW=%s\t%s' % (self.time, self.word, self.description))

def check_uniq_arc(arc, uniqarcs):
    if uniqarcs==[]:
        return False, None
    for a in uniqarcs:
        if (a.start.frame == arc.start.frame and a.end.frame == arc.end.frame and a.word == arc.word):
            return True, a
        else:
            return False, None

class Arc:
    def __init__ (self, line = None, nodes = None,
            startNode = None, endNode = None, description = None):
        if startNode == None:
            (start, end, rest) = tuple (line.split (None, 2))
            self.start = nodes [readInt (start, 'S')]
            self.end = nodes [readInt (end, 'E')]
            self.description = rest.strip()

            (aclike,lmlike) = tuple(rest.strip().split())
            self.aclike = readFloat (aclike, 'a')
            self.lmlike = readFloat (lmlike, 'l')
            self.acscore = LOGZERO
            self.lmscore = LOGZERO
            self.loss = 0.0
            self.transP = 0.0

            self.score = LOGZERO
            self.beststates = []

            #self.UniqArc = None                 # speed up lattice rescore
            #self.UniqArc_Processed = False      # speed up lattice rescore
        else:
            self.start = startNode
            self.end = endNode
            self.description = description

            (aclike,lmlike) = tuple(description.split())
            self.aclike = readFloat (aclike, 'a')
            self.lmlike = readFloat (lmlike, 'l')
            self.acscore = LOGZERO
            self.lmscore = LOGZERO
            self.loss = 0.0
            self.transP = 0.0

            self.score = LOGZERO
            self.beststates = []

            #self.UniqArc = None                 # speed up lattice rescore
            #self.UniqArc_Processed = False      # speed up lattice rescore
    
    def __str__ (self):
        return 'Arc (%s %s %s)' % (self.start, self.end, self.description)
    
    def rebind (self, mapping):
        '''Return a version of self with nodes remapped according to mapping.'''
        if self.start in mapping or self.end in mapping:
            if self.start in mapping:
                newStart = mapping [self.start]
            else:
                newStart = self.start
            if self.end in mapping:
                newEnd = mapping [self.end]
            else:
                newEnd = self.end
            return Arc (startNode = newStart, endNode = newEnd,
                description = self.description)
        else:
            return self
    
    def write (self, f, nodeIds):
        f.write ('S=%i\tE=%i\t%s' % (
            nodeIds [self.start], nodeIds [self.end], self.description))
    
    def getNodes (self):
        return set ([self.start, self.end])
    
class Lattice:
    def __init__ (self, lines):
        lines = iter (lines)
        line = lines.next()
        assert (line.strip() == 'VERSION=1.0')
        self.header = line
        for line in lines:                   # read the head
            if line.startswith ('N='):
                break
            self.header += line
        
        line = line.split()
        nodeNum = readInt (line[0], 'N')
        arcNum = readInt (line[1], 'L')
        
        print 'Num of Nodes:', nodeNum, '   Num of arcs:', arcNum
        self.nodesNum=nodeNum
        self.arcsNum=arcNum


        nodes = [] 
        for (index, line) in zip (xrange (nodeNum), lines):
            #print index, line
            (indexText, rest) = tuple (line.split (None, 1))
            assert (readInt (indexText, 'I') == index)
            #print Node
            #print rest
            nodes.append (Node (rest))
        
        self.start = nodes [0]
        self.end = nodes [-1]
        assert (self.start.word == '!NULL')
        assert (self.end.word == '!NULL')     
        #self.nodes=nodes                                    # lattice.nodes[0,...nodesNum]   each is a node structure
#        for node in nodes:
#            print node
        
        self.arcs = []
        for (index, line) in zip (xrange (arcNum), lines):
            (indexText, rest) = tuple (line.split (None, 1))
            assert (readInt (indexText, 'J') == index)
            self.arcs.append (Arc (rest, nodes))      

        # self.uniqarcs = []                                  # to speed up, because some arcs are duplicated, no need to do realignment
        # for arc in self.arcs:
        #     Uniq_or_not, uniq_arc = check_uniq_arc(arc, self.uniqarcs)
        #     if Uniq_or_not:                
        #         arc.UniqArc = uniq_arc
        #     else:
        #         self.uniqarcs.append(arc)
        #         arc.UniqArc = arc
        # print 'Num of Uniq arcs:', len(self.uniqarcs)

#        for arc in self.arcs:
#            print arc

        self.T = int((self.end.time+0.00001)*100)      # hard code, 10 ms per frame.  NOTE, the last frame are not exit. T=last_frame-1

        self.orderedNodes = self.getOrdered(self.start, self.arcs)  # whenever read a lattice, make its nodes in time order
        # self.arcsEndingIn will be create in self.getOrdered()
        assert (self.orderedNodes[0] == self.start)
    
    def add (self, lattice):
        # Map start and end nodes to the nodes in this lattice.
        mapping = { lattice.start : self.start, lattice.end : self.end }
        self.arcs += [arc.rebind (mapping) for arc in lattice.arcs]
    
    def getOrdered (self, start, arcs):
        addedNodes = set()
        orderedNodes = []
        queue = set([start])
        addedArcs = set()
#        orderedArcs = []
        arcsStartingIn = defaultdict(set)
        arcsEndingIn = defaultdict(set)
        for arc in arcs:
            arcsStartingIn[arc.start].add(arc)          # node as a key, arcsStartingIn={node:arcset, node:arcset}
            arcsEndingIn[arc.end].add(arc)
        while len (queue) != 0:
            for (time, node) in sorted ([(node.time, node) for node in queue]):
                # Check that all nodes from which there are arcs into this
                # are in addedNodes already.
                arcsEndingHere = arcsEndingIn[node]     # arcsEndingHere is just a arcset
                arcsEndingHere -= addedArcs
                if len (arcsEndingHere) == 0:
                    # The node can safely be added
                    addedNodes.add (node)
                    orderedNodes.append (node)
                    queue.remove (node)
                    # Add all arcs starting here and queue their end nodes.
                    for arc in arcsStartingIn [node]:
                        addedArcs.add (arc)
#                        orderedArcs.append (arc)
                        if not arc.end in addedNodes:
                            queue.add (arc.end)
                    break
                # Will loop forever if there is a cycle...!
        # This is true unless the graph is not connected.
#        assert (len (addedArcs) == len (arcs))
        self.arcsEndingIn=arcsEndingIn
        self.arcsStartingIn=arcsStartingIn
        return orderedNodes                             #  orderedNodes = lattice.getOrdered(lattice.start, lattice.arcs)

    def attachLoss(self, ref_states):

        assert(self.T == len(ref_states))
        for arc in self.arcs: 
            startframe = arc.start.frame
            endframe = arc.end.frame 
            arc_ref_states=ref_states[startframe:endframe]
            arc_ref_phone_perframe=[i.split('_')[0] for i in arc_ref_states]        # <------------ hard code
            arc.loss = len(arc_ref_states) - arc_ref_phone_perframe.count(arc.end.word)

    def clean_lat_search_info(self):
        self.start.bestpathscore = LOGZERO
        self.start.bestprevarc = None
        for arc in self.arcs:
            arc.acscore = LOGZERO
            arc.lmscore = LOGZERO
            arc.loss = 0.0
            arc.score = LOGZERO
            arc.beststates = []
        
            arc.end.bestpathscore = LOGZERO
            arc.end.bestprevarc = None

    def write (self, f):
        f.write (self.header)
        
        nodes = self.getOrdered (self.start, self.arcs)
        f.write ('N=%i\tL=%i\n' % (len (nodes), len (self.arcs)))
        
        nodeIds = {}
        for (index, node) in enumerate (nodes):
            f.write ('I=%i\t' % index)
            node.write (f)
            f.write ('\n')
            nodeIds [node] = index
        
        # The arcs in HTK lattices seem to be sorted by end node index.
        arcs = sorted ([(nodeIds [arc.end], arc) for arc in self.arcs])
        arcs = [arc for (index, arc) in arcs]
        
        for (index, arc) in enumerate (arcs):
            f.write ('J=%i\t' % index)
            arc.write (f, nodeIds)
            f.write ('\n')
    
    def sweepImpl (self, node, regime, memo, arcsEndingIn):
        incoming = []
        nodeValue = regime.nodeValue(node)
        for arc in arcsEndingIn[node]:
            if arc.start in memo:
                history = memo[arc.start]
            else:
                history = self.sweepImpl(arc.start, regime, memo, arcsEndingIn)
                memo[arc.start] = history
#            print 'Incoming: node', history,
            arcValue = regime.arcValue(arc)
#            print 'arc', arcValue,
            totalValue = regime.multiply(history, arcValue, nodeValue) 
#            print 'total', totalValue
            incoming.append(totalValue)
#        print 'Makes',
        result = regime.add(incoming)
#        print result
        return result
    
    def sweep (self, regime):
        # Start in self.start with probability 1
        memo = {self.start: regime.initialValue()}
        
        # Collect arcs by end node
        arcsEndingIn = defaultdict (set)
        for arc in self.arcs:
            arcsEndingIn [arc.end].add (arc)
        
        return self.sweepImpl(self.end, regime, memo, arcsEndingIn)

class SweepBase:
    def arcValue (self, arc):
        values = {}
        for (name, value) in extractValues (arc.description).iteritems():
            if name in self.names:
                values [name] = LogProb (exponent = float (value))
        assert (len (values) == len (self.names))
        return JointProbability (values)

class Forward (SweepBase):
    def __init__ (self, names):
        '''\param names names of features to be used'''
        self.names = names
        self.zero = JointProbability (dict ([(name, LogProb (0)) for name in names])) 
        self.one = JointProbability (dict ([(name, LogProb (1)) for name in names]))
    
    def initialValue (self):
        return self.one
    
    def nodeValue (self, node):
        return None
    
    def add (self, histories):
        return reduce (operator.add, histories, self.zero)
    
    def multiply (self, probability, arcValue, nodeValue):
        assert (nodeValue == None)
        return probability * arcValue

class Viterbi (SweepBase):
    def __init__ (self, names):
        '''\param names names of features to be used'''         # names=['l','a']
        self.names = names
        self.zero = JointProbability (dict ([(name, LogProb (0)) for name in names])) 
        self.one = JointProbability (dict ([(name, LogProb (1)) for name in names]))
    
    def initialValue (self):
        return ([], self.one)
    
    def nodeValue (self, node):
        return node.word
    
    def add (self, histories):
        if histories == []:
            return ([], self.zero)
            
        def mostLikely ((history1, probability1), (history2, probability2)):
            if probability1 > probability2:
                return (history1, probability1)
            else:
                return (history2, probability2)
        return reduce (mostLikely, histories)
    
    def multiply (self, (history, probability), arcValue, nodeValue):
        return (history + [nodeValue], probability * arcValue)


#################################################################################################################
#lattice = Lattice (open('D:\SSVM\svm-python-v204\debug_data\wlat\si1657.lat'))
#ordered_nodes=lattice.getOrdered(lattice.start, lattice.arcs)
#lattice.attachLoss(s_utt[1])

#################################################################################################################

def b_j_o_t(o_t, state_j, sm):

    frame_dim = len(o_t)
    weight = sm.w[state_j*frame_dim : (state_j+1)*frame_dim]                # sm.w[0:...]
    b_j_o_t_score = np.dot(o_t, weight)                                     # for LL feature, o_t is already in log domain
    return b_j_o_t_score

def maxtrix_softmax(mat):
    # thoetrically and empirically, the same result
    exp_mat=np.exp(mat)
    colSum=np.sum(exp_mat, axis=1)    
    den = np.dot( np.log(colSum).reshape(len(colSum),1), np.ones((1,mat.shape[1])) )
    logProbMatrix=  mat - den
    return logProbMatrix

def bjot_matrix(obs, sm):
    # matrix version, fast!
    bjot_mat = np.dot(obs, sm.acw_matrix)                                   # one line in obs is one frame
    #bjot_mat = maxtrix_softmax(bjot_mat)
    return bjot_mat  #bjot_matrix[t][j]

def bjot_matrix_OLD(obs, sm):
    bjot_mat=np.zeros((len(obs),sm.num_states), dtype=np.float)
    for t in xrange(len(obs)):
        for j in xrange(sm.num_states):
            bjot_mat[t,j]=b_j_o_t(obs[t], j, sm)

    return bjot_mat  #bjot_matrix[t][j]


def Update_transP(sparm, sm):                                               # do this in every iteration
    # np.log
    # transP[:, state_j] * sm.w[sm.num_states*frame_dim]
    for (k,v) in sparm.dict_MMF.items():
        sparm.dict_UpdatedMMF[k]=v * sm.w[sm.num_states*sm.num_features]          # v is log(tranP)

    #return transP * sm.w[sm.num_states*frame_dim]

def Viterbi_oneHMM(bjot, state_i2j, sm, transP):                             # NOTE obs is realdy bjot, state_i2j=[0,1,2] or [3,4,5] or [6,7,8] ... [180, 181, 182]
    #num_obs, frame_dim = obs.shape
    num_obs, total_num_states = bjot.shape                                   # obs is alreay bjot matrix
    num_states = transP.shape[0]                                            # num_states = 5
    assert ( len(state_i2j)+2 == transP.shape[0] )

    # initialize path costs going into each state, start with 0
    phi_t_j =  np.ones(num_states) * LOGZERO
    phi_tminus1_j = np.ones(num_states) * LOGZERO
    #phi_t_j[1] = b_j_o_t(obs[0], state_i2j[0], sm)       #+ np.log(transP[0, 1])
    
    phi_t_j[1] = bjot[0, state_i2j[0]]
    #phi_tminus1_j[1] = b_j_o_t(obs[0], state_i2j[0], sm)
    phi_tminus1_j[1] = bjot[0, state_i2j[0]]
    # initialize arrays to store best paths, 1 row for each ending state --- trilles
    paths = np.ones( (num_states, num_obs ), dtype=np.int) * -1                         # 1st column is nothing, just state labels
    #paths[:, 0] = np.arange(num_states)                                # read state id is [0,1,2] or [3,4,5] or [6,7,8], Here [0,1,2,3,4]
    paths[1,0] = 0
                                                                        # paths = [[-1,-1,-1,-1,-1,-1]
                                                                        #          [ 0,-1,-1,-1,-1,-1]    
                                                                        #          [-1,-1,-1,-1,-1,-1]
                                                                        #          [-1,-1,-1,-1,-1, X]    
                                                                        #          [-1,-1,-1,-1,-1,-1]]
    # start looping
    #for t, bjot in enumerate(bjot_arc[1:],1):                                 # skip 1st frame, must be the 1st emission state
    for t in xrange(1, num_obs):
        # for each obs, need to check for best path into each state
        for state_j in xrange(1,num_states-1):                          # [0,1,2,3,4] --> [1,2,3]
            # given observation, check prob of each path (column)       phi_t(j) = max_{i} { phi_t-1(0,...i,..4) + b_j(o_t) + [a_0j, ...,a_ij, ..., a_4j] }
            # state_ind_in_TransP = state_j - (state_i2j[0]-1)
            #print 'time ', t, 'state', state_j
            real_state_j = (state_i2j[0]-1) + state_j
            phi_t_j_from = phi_tminus1_j + bjot[t,real_state_j] + transP[:, state_j]
            # transP already scaled and log
            # transP[:, state_j] * sm.w[sm.num_states*frame_dim]                         
            # b_j_o_t(o_t, real_state_j, sm)
            
            # check for largest score                        
            best_prev_ind = np.argmax(phi_t_j_from)
            phi_t_j[state_j] = phi_t_j_from[best_prev_ind]            
            #print 'phi_t_j_from ', phi_t_j_from
            #print best_prev_ind            
            # save the path with a higher prob and score
            paths[state_j,t] = best_prev_ind                  # note that path[:,0] is for states, tracing from 1...T, 
        phi_tminus1_j[:] = phi_t_j                            # VERY IMPORTANT, MUST HAVE[:], otherwise will only pass the pointer
        #print 'phi_t_j ', phi_t_j
            
    #print 'paths ', paths
    # we now have a best stuff going into each path, find the best score
    assert(state_j==num_states-2)                                      # now, state_j should be the last state, t is the last
    best_score = phi_tminus1_j[num_states-2] + transP[-2,-1]           # VERY IMPORTANT, np.log already, DO NOT FORGET exit transp
    real_state_j = (state_i2j[0]-1) + state_j
    best_state_seq = [real_state_j]                                    # now, state_j should be the last state
                           

    best_prev_ind = state_j

    for t in xrange(num_obs-1,0,-1):                                    # t= T,...1,   skip the first frame 0 
        best_prev_ind = paths[best_prev_ind,t]                          # best_prev_ind should start from 3 
        real_state_j = (state_i2j[0]-1) + best_prev_ind
        best_state_seq.insert(0, real_state_j)   
    #assert(best_prev_ind ==1 )                                         # (j, t=1) must point to (j=1, t=0)
    assert(len(best_state_seq)==num_obs)
    #print best_state_seq[0], state_i2j[0]
    #print bjot.shape
    assert(best_state_seq[0]==state_i2j[0])
    assert(best_state_seq[-1]==state_i2j[-1])


    # done, get out.
    return best_score, best_state_seq


def loss_seq(str1, str2):
    assert(len(str1)==len(str2))
    loss=0
    for i in xrange(len(str1)):
        if str1[i]!=str2[i]:
            loss=loss+1
    return loss



def loss_frame(str1, str2):    
    if str1==str2:
        return 0
    else: 
        return 1


def Viterbi_oneHMM_withLoss(bjot, ref_states, state_i2j, sm, transP):        # state_i2j=[0,1,2] or [3,4,5] or [6,7,8]   
    num_obs, total_num_states = bjot.shape                                         # obs is alreay bjot matrix
    num_states = transP.shape[0]                                            # num_states = 5
    assert ( len(state_i2j)+2 == transP.shape[0] )

    # initialize path costs going into each state, start with 0
    phi_t_j =  np.ones(num_states) * LOGZERO
    phi_tminus1_j = np.ones(num_states) * LOGZERO
    #phi_t_j[1] = b_j_o_t(obs[0], state_i2j[0], sm)       #+ np.log(transP[0, 1])
    phi_t_j[1] = bjot[0, state_i2j[0]]
    #phi_tminus1_j[1] = b_j_o_t(obs[0], state_i2j[0], sm)
    phi_tminus1_j[1] = bjot[0, state_i2j[0]]
    # initialize arrays to store best paths, 1 row for each ending state --- trilles
    paths = np.ones( (num_states, num_obs ), dtype=np.int) * -1                         # 1st column is nothing, just state labels
    #paths[:, 0] = np.arange(num_states)                                # read state id is [0,1,2] or [3,4,5] or [6,7,8], Here [0,1,2,3,4]
    paths[1,0] = 0
                                                                        # paths = [[-1,-1,-1,-1,-1,-1]
                                                                        #          [ 0,-1,-1,-1,-1,-1]    
                                                                        #          [-1,-1,-1,-1,-1,-1]
                                                                        #          [-1,-1,-1,-1,-1, X]    
                                                                        #          [-1,-1,-1,-1,-1,-1]]
    # start looping
    #for t, bjot in enumerate(bjot_arc[1:],1):                                 # skip 1st frame, must be the 1st emission state
    for t in xrange(1, num_obs):
        # for each obs, need to check for best path into each state
        for state_j in xrange(1,num_states-1):                          # [0,1,2,3,4] --> [1,2,3]
            # given observation, check prob of each path (column)       phi_t(j) = max_{i} { phi_t-1(0,...i,..4) + b_j(o_t) + [a_0j, ...,a_ij, ..., a_4j] }
            # state_ind_in_TransP = state_j - (state_i2j[0]-1)
            #print 'time ', t, 'state', state_j
            real_state_j = (state_i2j[0]-1) + state_j
            phi_t_j_from = phi_tminus1_j + bjot[t,real_state_j] + transP[:, state_j] + loss_frame(real_state_j, ref_states[t])
                         # transP already scaled and log
                         # transP[:, state_j] * sm.w[sm.num_states*frame_dim]                         
                         # b_j_o_t(o_t, real_state_j, sm)
            # check for largest score                        
            best_prev_ind = np.argmax(phi_t_j_from)
            phi_t_j[state_j] = phi_t_j_from[best_prev_ind]            
            #print 'phi_t_j_from ', phi_t_j_from
            #print best_prev_ind            
            # save the path with a higher prob and score
            paths[state_j,t] = best_prev_ind                  # note that path[:,0] is for states, tracing from 1...T, 
        phi_tminus1_j[:] = phi_t_j                            # VERY IMPORTANT, MUST HAVE[:], otherwise will only pass the pointer
        #print 'phi_t_j ', phi_t_j
            
    #print 'paths ', paths
    # we now have a best stuff going into each path, find the best score
    assert(state_j==num_states-2)                                      # now, state_j should be the last state, t is the last
    best_score = phi_tminus1_j[num_states-2] + transP[-2,-1]           # VERY IMPORTANT, np.log already, DO NOT FORGET exit transp
    real_state_j = (state_i2j[0]-1) + state_j
    best_state_seq = [real_state_j]                                    # now, state_j should be the last state
                           

    best_prev_ind = state_j
    for t in xrange(num_obs-1,0,-1):                                    # t= T,...1,   skip the first frame 0 
        best_prev_ind = paths[best_prev_ind,t]                          # best_prev_ind should start from 3 
        real_state_j = (state_i2j[0]-1) + best_prev_ind
        best_state_seq.insert(0, real_state_j)   
    #assert(best_prev_ind ==1 )                                         # (j, t=1) must point to (j=1, t=0)
    assert(len(best_state_seq)==num_obs)
    assert(best_state_seq[0]==state_i2j[0])
    assert(best_state_seq[-1]==state_i2j[-1])


    # transP of best state sequences
    #best_states_transP = States_TRANSP(best_state_seq, transP)

    # done, get out.
    arc_loss = loss_seq(ref_states, best_state_seq)
    best_acscore = best_score - arc_loss
    return best_acscore, best_state_seq, arc_loss



def lmscore(arc, sm, sparm):

    if (sparm.Featype=='oneScalar'):
        lm_score=arc.lmlike*sm.w[sm.num_states*sm.num_features+1]
    if (sparm.Featype=='lmWeights'):
        wrd_i = sparm.dict_model_id[arc.start.word]             # id from 0
        wrd_j = sparm.dict_model_id[arc.end.word]
        lm_score=arc.lmlike*sm.w[num_states*frame_dim+1 + wrd_i*61+wrd_j]       # <-------------------- hard code len(sparm.dict_model_id)
    if (sparm.Featype=='NOlmFea'):
        lm_score=arc.lmlike*sparm.lmscalar

    return lm_score



#############################################################################################################################################

def LArcScore(arc, bjot, ref_states, sm, sparm):                 # Acrscore = Viterbi_acscore(bjot+TransScore) + lmscore + loss

    startframe = arc.start.frame                                # int((arc.start.time+0.00001) * 100)
    endframe = arc.end.frame                                    # int((arc.end.time+0.00001) * 100)

    transP =  sparm.dict_UpdatedMMF[arc.end.word]           
    state_i2j = sparm.dict_phn_si2j[arc.end.word]
    bjot_arc = bjot[startframe:endframe]             # Note that: Arc_endframe is belongs to next arc

    if (ref_states==''):
        arc.acscore, arc.beststates = Viterbi_oneHMM(bjot_arc, state_i2j, sm, transP)   # arc.loss <---- lattice.attachLoss(s_utt[1])

        arc.lmscore = lmscore(arc, sm, sparm)
        return arc.acscore + arc.lmscore
    else:
        arc_ref_states = ref_states[startframe:endframe] 
        arc.acscore, arc.beststates, arc.loss = Viterbi_oneHMM_withLoss(bjot_arc, arc_ref_states, state_i2j, sm, transP)     #arc.aclike, arc.lmlike, arc.score
    
        arc.lmscore = lmscore(arc, sm, sparm)
  
        return arc.acscore + arc.lmscore + arc.loss


# def lattice_initial(lat, example):

#     x=example[0][1]
#     y=example[1][1]
#     assert(len(y) == lat.T)
#     ref_states=[]
#     for state in y:
#         ref_states.append(dict_state[state])
#     loss_on_arc = loss_states(ref_states,arc)

#     # read lattice function above

#     return lat


def debug_arc(arc,obs):
    print "WARNING arc.acscore - arc.aclike = ", (arc.acscore-arc.aclike), 'arc.start.frame ', arc.start.frame, 'arc.end.frame ', arc.end.frame, 'arc.end.word', arc.end.word, '\n'
    print arc.beststates
    obsseq=[]
    for t in xrange(arc.start.frame, arc.end.frame):
        j=arc.beststates[t-arc.start.frame]
        obsseq.append(obs[t][j])        
    print obsseq, arc.aclike, '\n'

def lattice_search(lat, obs, ref_states, sm, sparm):

    tmpscore = LOGZERO    
    #obs=example[0][1]
    #ref_states=example[1][1]

    bjot=bjot_matrix(obs, sm)  

    orderedNodes=lat.getOrdered(lat.start, lat.arcs)
    assert (orderedNodes[0] == lat.start)
    
    arcsStartingIn = defaultdict(set)
    arcsEndingIn = defaultdict(set)
    for arc in lat.arcs:
        arcsStartingIn[arc.start].add(arc)          # node as a key, arcsStartingIn={node:arcset, node:arcset}
        arcsEndingIn[arc.end].add(arc)

    # search best path for each node, in time order    
    lat.start.bestpathscore = 0.0
    for node in orderedNodes:
        for arc in arcsStartingIn[node]:
            assert (arc.start == node)
            #startframe = int((arc.start.time+0.00001) * 100)
            #endframe = int((arc.end.time+0.00001) * 100)

            #arc_ref_states = ref_states[startframe:endframe]
            #arc_obs=obs[startframe:endframe]
            if (arc.end.word != '!NULL' and arc.end.word != '!ENTER' and arc.end.word != '!END'):
                tmpscore = arc.start.bestpathscore + LArcScore(arc, bjot, ref_states, sm, sparm)   # WHOLE obs/states #arc.aclike + 3.0* arc.lmlike 
            else:
                tmpscore = arc.start.bestpathscore + arc.lmlike * sm.w[sm.num_states*sm.num_features+1] # arc.lm=0.0 anyway
            if (tmpscore > arc.end.bestpathscore):
                arc.end.bestpathscore=tmpscore
                arc.end.bestprevarc=arc

    # backtrace    
    node = lat.end                  # be careful, the last node is !NULL    
    comp_words =[]
    COMP_STATES = []    
    fea_LM = 0.0
    fea_TRANSP = 0.0
    loss_utt = 0.0
    while (node != lat.start):
        if (node.word != "!NULL" and node.word != "!ENTER" and node.word != "!EXIT"):
            prev_arc = node.bestprevarc
            arc_states = prev_arc.beststates
                  
            comp_words.insert(0, node.word)            
            COMP_STATES.insert(0, arc_states)            

            fea_LM = fea_LM + prev_arc.lmscore
            #fea_TRANSP = fea_TRANSP + prev_arc.transP
            loss_utt = loss_utt + prev_arc.loss

        node = node.bestprevarc.start
    
    return COMP_STATES, loss_utt, fea_LM     


def lattice_search_decode(lat, obs, sm, sparm):

    tmpscore = LOGZERO    
    #obs=example[0][1]
    #ref_states=example[1][1]
    bjot = bjot_matrix(obs, sm)                           # time-consuming point, OLD:1.781s, NOW:0.016s

    
    orderedNodes = lat.orderedNodes                       # lat.getOrdered(lat.start, lat.arcs) time consume: 0.203s   

    
    arcsStartingIn = lat.arcsStartingIn 
    # timestart = time.clock()
    # for arc in lat.arcs:
    #     arcsStartingIn[arc.start].add(arc)              # node as a key, arcsStartingIn={node:arcset, node:arcset}.  # time consume: 0.016s
    #     arcsEndingIn[arc.end].add(arc)
    # timeend = time.clock()
    # print timeend-timestart

    
    # search best path for each node, in time order    
    lat.start.bestpathscore = 0.0
    for node in orderedNodes:
        for arc in arcsStartingIn[node]:
            assert (arc.start == node)
            #startframe = int((arc.start.time+0.00001) * 100)
            #endframe = int((arc.end.time+0.00001) * 100)
            #arc_ref_states = ref_states[startframe:endframe]
            #arc_obs=obs[startframe:endframe]           
            if (arc.end.word != '!NULL' and arc.end.word != '!ENTER' and arc.end.word != '!EXIT'):
                tmpscore = arc.start.bestpathscore + LArcScore(arc, bjot, '', sm, sparm)   # WHOLE obs/states     # arc.aclike + 3.0* arc.lmlike   # LArcScore(arc, obs, '', sm, sparm)             
                
                #print arc.acscore, arc.aclike, LArcScore(arc, obs, '', sm, sparm)
                #if ( (arc.acscore - arc.aclike) < -0.3):
                #    debug_arc(arc,obs)
            else:
                tmpscore = arc.start.bestpathscore + arc.lmlike * sm.w[sm.num_states*sm.num_features+1] # arc.lm=0.0 anyway
            if (tmpscore > arc.end.bestpathscore):
                arc.end.bestpathscore=tmpscore
                arc.end.bestprevarc=arc

    # backtrace
    #COMP_STATES = []    
    node = lat.end                  # be careful, the last node is !NULL
    #comp_words = []
    best_arcs = []
    while (node != lat.start):
        if (node.word != "!NULL" and node.word != "!ENTER" and node.word != "!EXIT"):
            prev_arc = node.bestprevarc
            #arc_states = prev_arc.beststates
                    
            #comp_words.insert(0, node.word)
            best_arcs.insert(0, prev_arc)
            #print node.word
            #print node.time
            #print node.bestpathscore
        node = node.bestprevarc.start

    return best_arcs
 #print lattice.getOrdered(lattice.start, lattice.arcs)


#def lattice_state_marking(lat, obs, sm, sparm):
    