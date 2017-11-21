#!/bin/python

# Date created: Nov 5 2017

# This serves as the main script for agglomerative clustering for speaker diarization using the information bottleneck criterion.
# There are additional scripts planned:
#	functions.py: Various mathematical functions - entropy, mutual information, JS-divergence, etc
#	uniform_segment.py: Uniformly segment the speech regions from a session and extract features from them
#	infoBottlenech.py: (This script) Perform the speaker diarization
#	reAlign.py: Perform fine-tuning of the speaker boundaries using the viterbi criterion

import sys
import warnings
import numpy as np
import pandas as pd
from functions import *
from scipy.cluster.hierarchy import fcluster

def run_synthetic_example():
	# Used mostly while algorithm development
	np.random.seed(1000)
	np.set_printoptions(precision=5)
	np.core.arrayprint._line_width = 160
	N,d,P,Q = 60,3,4,5
	X = pd.DataFrame(np.random.rand(N,d))
	Y = np.random.rand(P,Q)
	p_y_x = np.empty((N,P))
	for i in range(N/2):	
		p_y_x[i] = [0.91,0.03,0.03,0.03]
	for i in range(N/2,N):	
		p_y_x[i] = [0.03,0.03,0.03,0.91]
	for i in range(N/4,N/2):	
		p_y_x[i] = [0.03,0.91,0.03,0.03]
	for i in range(N/2,3*N/4):	
		p_y_x[i] = [0.03,0.03,0.91,0.03]

	cluster(p_y_x, 2, 0.5,1)

def run_example():
	# Using a clip from AMI corpus consisting of 1 male speaker and 1 female speaker

	segLen,frameRate = 2,100
	np.random.seed(1000)
	np.set_printoptions(precision=5)
	np.core.arrayprint._line_width = 160
	wavFile = 'ami_test_file/ES2002_test.wav'
	vad = doVAD(wavFile)
	mfcc,GMM = createGMM(wavFile,32,vad=vad)
	p_y_x = uniformSegmentsAndPosterior(mfcc,GMM,segLen=segLen,frameRate=frameRate)
	p_y_x = softmax(p_y_x)
	Z,C = cluster(p_y_x,2, 10,0)

	clust = fcluster(Z,2,criterion='maxclust')
	frameClust = convertDecisionsSegToFrame(clust, segLen, frameRate, mfcc.shape[0])
	pass1hyp = -1*np.ones(len(vad))
	pass1hyp[vad] = frameClust
	np.savetxt('pass1hypSpkr.txt',pass1hyp,fmt="%3.2f")

	# By enforcing a minimum segment (block) size
	frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=25,numMix=16)
	pass2hyp = -1*np.ones(len(vad))
	pass2hyp[vad] = frameClust
	np.savetxt('pass2hypSpkr.txt',pass2hyp,fmt="%3.2f")

	# Trying with multiple passes
	frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=20,numMix=16)
	pass3hyp = -1*np.ones(len(vad))
	pass3hyp[vad] = frameClust
	np.savetxt('pass3hypSpkr.txt',pass3hyp,fmt="%3.2f")

	frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=10,numMix=16)
	pass4hyp = -1*np.ones(len(vad))
	pass4hyp[vad] = frameClust
	np.savetxt('pass4hypSpkr.txt',pass4hyp,fmt="%3.2f")

run_example()




