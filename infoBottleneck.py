#!/bin/python

# Date created: Nov 5 2017

# This serves as the main script for agglomerative clustering for speaker diarization using the information bottleneck criterion.
# This script includes one synthetic example, two real examples - clips from the AMI meeting corpus, and a function to perform diarization on a single file given the VAD

# All relevant functions for performing agglomerative clustering ( computing mutual information, JS-divergence, KL-distance, etc.) functions for feature extraction, GMM training, Viterbi re-alignment, etc. are included in functions.py. 

import sys
import warnings
import numpy as np
from functions import *
from scipy.cluster.hierarchy import fcluster

def run_synthetic_example():
	# Used mostly during  algorithm development

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

	cluster(p_y_x, 2, 10,1)

def run_example():
	# Using a clip from AMI corpus consisting of 1 male speaker and 1 female speaker

	segLen,frameRate,numMix = 2,100,32
	np.random.seed(1000)
	np.set_printoptions(precision=5)
	np.core.arrayprint._line_width = 160
	wavFile = 'ami_test_file/EN2005_test.wav'
	vad = doVAD(wavFile,frameRate)
	np.savetxt('vad.txt',vad)
	mfcc,GMM = trainGMMScipy(wavFile, numMix, frameRate, vad=vad)
	p_y_x = uniformSegmentsAndPosterior(mfcc,GMM,segLen=segLen,frameRate=frameRate)
	p_y_x[p_y_x<1e-10] = 1e-10
	Z,C = cluster(p_y_x,2, 10,0)

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



def singleFile(wavFile, vadFile, rttmFile, numClusters):	
	# Takes a wavefile as input, and outputs the diarization results in rttm format

	segLen,frameRate = 2,50
	np.random.seed(1000)
	vad = np.loadtxt(vadFile).astype('bool')
	vad = np.interp(np.linspace(0,len(vad),int(len(vad)*frameRate/100.0)),np.arange(len(vad)),vad).astype('bool')

	# Training using scipy GMM's
	# mfcc,vad,p_y_x = trainGMMScipy(wavFile, 'models/scipy_64mix_13mfcc.sav', frameRate, segLen, vad=None, localGMM=True, numMix=128)

	# ..... OR ..... 

	# Training using kaldi GMM's
	mfcc,vad,p_y_x = extractFeatAndPostKaldi(wavFile, 'models/ami_kaldi_64mix_19mfcc_nodel.mdl', frameRate, segLen, '/home/manoj/kaldi/', vad=vad, localGMM=True, numMix=128)

	p_y_x[p_y_x<1e-10] = 1e-10
	Z,C = cluster(p_y_x,2,10,0)

	clust = fcluster(Z,numClusters,criterion='maxclust')
	frameClust = convertDecisionsSegToFrame(clust, segLen, frameRate, mfcc.shape[0])
	pass1hyp = -1*np.ones(len(vad))
	pass1hyp[vad] = frameClust

	frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=25,numMix=16)
	pass2hyp = -1*np.ones(len(vad))
	pass2hyp[vad] = frameClust
	if len(np.unique(pass2hyp)) < numClusters+1:
		writeRttmFile(pass1hyp, frameRate, wavFile, rttmFile)
		return

	writeRttmFile(pass2hyp, frameRate, wavFile, rttmFile)

	# If required, re-alignments can be repeated. 
#	frameClust = viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=20,numMix=16)
#	pass3hyp = -1*np.ones(len(vad))
#	pass3hyp[vad] = frameClust
#	if len(np.unique(pass3hyp)) < numClusters+1:
#		writeRttmFile(pass2hyp, frameRate, wavFile, rttmFile)
#		return



#============ MAIN FUNCTION ==========
if __name__ == '__main__':
	if len(sys.argv)!=5:
		print("Usage: "+sys.argv[0]+" <audioFile> <vadFile> <output rttm file> <numClusters> ")
		sys.exit(1)

	singleFile(sys.argv[1],sys.argv[2],sys.argv[3], int(sys.argv[4]))
