#!/bin/python

# Date created: Dec 20 2017

# Perform diarization on a clip from the AMI corpus consisting of 1 male + 1 female speaker
# (ES2002_test) or 2 female speakers (EN2005a)

import numpy as np
from functions import *
from scipy.cluster.hierarchy import fcluster

segLen,frameRate,numMix = 2,50,128
np.random.seed(1000)
np.set_printoptions(precision=5)  
wavFile = 'ami_test_file/EN2005_test.wav'
wavData,_ = librosa.load(wavFile,sr=16000)
vad = doVADWithSklearn(wavData,frameRate)
np.savetxt('vad.txt',vad)
mfcc,vad,p_y_x = trainGMMWithSklearn(wavFile, None, frameRate, segLen, vad, 1, numMix)
p_y_x[p_y_x<1e-10] = 1e-10
Z,C = cluster(p_y_x,10,0)

clust = fcluster(Z,2,criterion='maxclust')
frameClust = convertDecisionsSegToFrame(clust, segLen, frameRate, mfcc.shape[0])
pass1hyp = -1*np.ones(len(vad))
pass1hyp[vad] = frameClust
writeRttmFile(pass1hyp, frameRate, wavFile, 'pass1hypSpkr.rttm')

# By enforcing a minimum segment (block) size
frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=25,numMix=16)
pass2hyp = -1*np.ones(len(vad))
pass2hyp[vad] = frameClust
writeRttmFile(pass2hyp, frameRate, wavFile, 'pass2hypSpkr.rttm')
