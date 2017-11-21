#!/bin/python

# Date created: Nov 5 2017

# Defines various functions necessary for IB-based speaker clustering

import os
import warnings
import kaldi_io
import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage.filters import *
from scipy.signal import *
from sklearn.mixture import *

# Get p_x_y_joint from p_y_x
def getJointFromConditional(p_y_x, p_x = 0):
	# If p(x) is not provided, it's assumed uniform
	N,P = np.shape(p_y_x)
	p_y = np.zeros(P)
	p_x_y_joint = np.zeros((N,P))
	if p_x == 0:
		p_x = (1.0/N)*np.ones(N)

	for j in range(P):
		for i in range(N):
			p_x_y_joint[i,j] = p_x_y_joint[i,j] + p_x[i]*p_y_x[i,j]
		 
	p_y = np.sum(p_x_y_joint,0)
	return p_x_y_joint, p_y, p_x

# Compute Jensen-Shannon divergence between p_1(x|i) and p_1(x|j)
def JS_div(p_z_c,p_c,i,j,call):
	# Compute the Jenson-Shannon divergence for the two variables with distributions p_z_c(z|i) and p_z_c(z|j)
	A,B = np.shape(p_z_c)
	pie_i, pie_j = p_c[i]/(p_c[i] + p_c[j]), p_c[j]/(p_c[i] + p_c[j])
	q_z = pie_i*p_z_c[i,:] + pie_j*p_z_c[j,:]
	return pie_i*KL_div(p_z_c[i,:],q_z,call,pie_i,pie_j) + pie_j*KL_div(p_z_c[j,:],q_z,call,pie_i,pie_j)

# Compute Kulback-Leibler divergence between two random variables
def KL_div(p_x, p_y,call,pie_i,pie_j):
	warnings.filterwarnings('error')
	if len(p_x) != len(p_y):
		raise ValueError("Dim mismatch at KL_div")
	mysum = 0
	for j in range(len(p_x)):

		if p_x[j] >0 and p_y[j] > 0:
			try:
				mysum += p_x[j]*np.log2(p_x[j]/p_y[j])
			except Warning:
				print ("Nr: "+str(p_x[j])+" dr: "+str(p_y[j])+" call: "+str(call)+" ("+str(pie_i)+","+str(pie_j)+")")
				print 'Warning was raised as an exception!'

		elif p_x[j]==0:
			mysum += 0
		else:
			mysum = float("inf")

	return mysum

def doVAD(wavFile):
	# Classifies frames into voiced/unvoiced, using Kaldi's pov feature
	# Inputs:
	# wavFile: Full path to wave file
	#	A string
	with open("temp.scp","w") as input_scp:
		input_scp.write("temp %s" % wavFile)
	os.system('/home/manoj/kaldi/src/featbin/compute-kaldi-pitch-feats --min-f0=40 --max-f0=600 scp:temp.scp ark:pitch.ark')
	for key,mat in kaldi_io.read_mat_ark('pitch.ark'):
		nccf = mat[:,0]
	l = -5.2 + 5.4*np.exp(7.5*(abs(nccf)-1)) + 4.8*abs(nccf) -2*np.exp(-10*abs(nccf)) + 4.2*np.exp(20*(abs(nccf)-1))
	p = gaussian_filter1d(1./(1 + np.exp(-l)),np.sqrt(10))
	vad = p>0.1
	return vad
	return medfilt(vad,11)
	



def createGMM(wavFile, numMix,vad=None):
	# Given an audio file, train a GMM using the EM algorithm
	# Inputs:
	# wavFile: Full path to wave file
	#	A string
	# numMix: number of mixture in the GMM
	#	A scalar
	# vad: 	Voiced activity decisions at frame level (Currently, frame rate = 100)
	#	A numpy logical array

	with open("temp.scp","w") as input_scp:
		input_scp.write("temp %s" % wavFile)
	os.system('/home/manoj/kaldi/src/featbin/compute-mfcc-feats --use-energy=false --num-ceps=19 --subtract-mean=true scp:temp.scp ark:out.ark')
#	os.system('/home/manoj/kaldi/src/featbin/compute-plp-feats --use-energy=false  scp:temp.scp ark:out.ark')
	for key,mat in kaldi_io.read_mat_ark('out.ark'):
		if vad is not None:
			mfcc = mat[vad,:]
		else:	
			mfcc = mat			
	GMM = GaussianMixture(n_components=numMix,covariance_type='diag').fit(mfcc)
	os.system("rm -f temp.scp out.ark")
	return mfcc, GMM
	

def uniformSegmentsAndPosterior(mfcc,GMM,segLen=2,frameRate=100):
	# Given an audio file and GMM, computes posteriors (p_y_x) using the GMM after uniformly segmenting 
	# Inputs:
	# mfcc:	Feature stream for the entire wave file
	#	A numpy array of size [n_frames,n_dim]
	# GMM: 
	#	An sklearn GMM object
	# segLen: Length of segment (in seconds)
	#	A scalar
	# frameRate: Number of frames per seconds
	#	A scalar

	var_floor = 1e-5
	segLikes = []
	segSize = frameRate*segLen
	for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
		startI = segI*segSize
		endI = (segI+1)*segSize
		if endI > mfcc.shape[0]:
			endI = mfcc.shape[0]-1
		seg = mfcc[startI:endI,:]
		compLikes = np.sum(GMM.predict_proba(seg),0)
		segLikes.append(compLikes/seg.shape[0])
	return np.asarray(segLikes)

def convertDecisionsSegToFrame(clust, segLen, frameRate, numFrames):
	# Convert cluster assignemnts from segment-level to frame-level
	# Inputs:
	# clust: Speaker hypothesis values at segment level 
	#	A numpy array of length N 
	# segLen: Length of segment (in seconds)
	#	A scalar
	# frameRate: Number of frames per seconds
	#	A scalar
	# numFrames: total number of frames	
	frameClust = np.zeros(numFrames)
	for clustI in range(len(clust)-1):
		frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
	frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
	return frameClust

def softmax(p_y_x):
	# Computes softmax for p(y|x)
	for i in range(p_y_x.shape[0]):
		arr = p_y_x[i,:] - max(p_y_x[i,:])
		p_y_x[i,:] = np.exp(arr)/sum(np.exp(arr))
	return p_y_x

def cluster(p_y_x, numCluster, beta, visual):
	# The main clustering function
	# Inputs:
	# p_y_x: Conditional probability p(y|x)
	#	A numpy array of size [N,P]
	# beta: Tradeoff parameter in the IB objective
	# 	A scalar
	# visual: Print dendrogram
	#
	#
	# Objective: Min (1/beta)*I(X,C) - I(Y,C)
	#
	#
	# Outputs:
	# C: Cluster assignment; an m-partitiion of X, 1 <= m <= |X| 
	#	A numpy array of size [N,1]
	#
	#
	# Some relevant variables
	# X: Features at segment level
	#	A pandas dataframe of length N, with each data object of size [ni,d]		
	# Y: Relevance variable, typically components from a GMM
	#	A numpy array of size [P,Q]


	print("Performing agglomerative clustering using IB objective...")
	N,P = np.shape(p_y_x)	
	np.random.seed(1000)
	p_c = np.empty(N)	
	p_y_c = np.empty((N,P))	# p(y|c), NOT p(y,c)
	p_c_x = np.zeros((N,N))
	p_x_c = np.zeros((N,N))
	p_x_y_joint = getJointFromConditional(p_y_x)
	delta_F = np.zeros((N,N))
	N_init = N

	# Initialization
#	print("Initialization...")
	C = range(N)
	for i in range(N):
		p_c[i] = 1.0/N
		p_c_x[i,i] = 1.0
		p_x_c[i,i] = 1.0
		for j in range(P):
			p_y_c[i,j] = p_y_x[i,j]

	for i in range(N):
		for j in range(i):
			delta_F[i,j] = (p_c[i] + p_c[j])*(JS_div(p_y_c,p_c,i,j,1) - (1/beta)*JS_div(p_x_c,p_c,i,j,2))
		for j in range(i,N):
			delta_F[i,j] = float("inf")

	
#	print p_y_c
#	print p_c_x
#	print p_x_c
	# Clustering
	max_clust_ind = max(C)
	Z = np.empty((max_clust_ind,4))
	curr_val = 0
	iterIndex = 0
	while len(np.unique(C))>2:
#		print("Number of clusters = "+str(N))
#		print "delta_F"
#		print delta_F
#		print("Performing one iteration of clustering..")
		[i_opt,j_opt] = np.unravel_index(np.argmin(delta_F), delta_F.shape)
#		print ("Optimal indices: ("+str(i_opt)+","+str(j_opt)+")")
		curr_val += abs(np.min(delta_F))
		Z[iterIndex] = [C[i_opt],C[j_opt],curr_val,2]
		iterIndex += 1
#		Z.append([C[i_opt],C[j_opt],curr_val,2])

		# Merge C[i_opt], C[j_opt]
			# Copy unmerged values in p(y|c) and p(c|x)
		C_new = []
		p_y_c_new = np.empty((N,P))
		p_c_x_new = np.empty((N_init,N))
		p_c_new = []
		for i in range(N):
			if i!=i_opt and i!=j_opt:
				C_new.append(C[i])
				p_c_new.append(p_c[i])
				for j in range(P):
					p_y_c_new[i,j] = p_y_c[i,j]
				for i2 in range(N_init):
					p_c_x_new[i2,i] = p_c_x[i2,i]

		p_y_c_new = np.delete(p_y_c_new,(i_opt,j_opt),0)
		p_c_x_new = np.delete(p_c_x_new,(i_opt,j_opt),1)
#		print("After removing the merging columns...")
		delta_F = np.delete(np.delete(delta_F,(i_opt,j_opt),0),(i_opt,j_opt),1)
#		print(delta_F)

			# Update p(y|c)
		C_new.append(max_clust_ind+1)
		temp1 = np.zeros(P)
		for j in range(P):
			temp1[j] = (p_y_c[i_opt,j]*p_c[i_opt] + p_y_c[j_opt,j]*p_c[j_opt])/(p_c[i_opt] + p_c[j_opt])
		p_y_c_new = np.vstack((p_y_c_new,temp1))

			# Update p(c|x)
		temp2 = np.zeros(N_init)
		for i in range(N):
			if i!=i_opt and i!=j_opt:
				temp2[i] = 0
			else:
				temp2[i] = 1
		p_c_x_new = np.concatenate((p_c_x_new,np.reshape(temp2,(len(temp2),1))),1)
		
			# Update p(c)
		p_c_new.append(p_c[i_opt] + p_c[j_opt])
		max_clust_ind += 1
#		N -= 1
		C = C_new
		p_y_c = p_y_c_new
		p_c_x = p_c_x_new
		p_c = np.asarray(p_c_new)
		p_x_c = np.empty((N-1,N_init))	# this should be N-1,N_init

		# Update p(x|c)
		for i in range(N-1):
			for i2 in range(N_init):
				p_x_c[i,i2] = p_c_x[i2,i]/(N_init*p_c[i])

		N -= 1
		p_y_c[p_y_c<10e-10] = 0.
		p_c_x[p_c_x<10e-10] = 0.
		p_x_c[p_x_c<10e-10] = 0.
		p_c[p_c<10e-10] = 0.


		# [New] Update delta_F
		# Add a row 
		newrow = np.zeros(N-1)
		for i in range(N-1):
			newrow[i] = (p_c[-1] + p_c[i])*(JS_div(p_y_c,p_c,i,len(p_c)-1,1) - (1/beta)*JS_div(p_x_c,p_c,i,len(p_c)-1,2))
		# Add a column of "inf"
		newcol = float("inf")*np.ones(N)
		delta_F = np.concatenate((np.vstack((delta_F,newrow)),np.reshape(newcol,(len(newcol),1))),1)


		# [Old] Recompute delta_F - This can be optimized by recomputing only for the new indices! - Look above
#		delta_F = np.zeros((N,N))
#		for i in range(N):
#			for j in range(i):
#				delta_F[i,j] = (p_c[i] + p_c[j])*(JS_div(p_y_c,p_c,i,j,1) - (1/beta)*JS_div(p_x_c,p_c,i,j,2))
#			for j in range(i,N):
#				delta_F[i,j] = float("inf")

#		print p_y_c.shape
#		print p_c_x.shape
#		print p_x_c.shape
#		print p_c.shape		
#		sys.exit()

				
#		print "p_y_c:"
#		print p_y_c
#		print "p_c_x:"
#		print p_c_x
#		print "p_x_c:"
#		print p_x_c
#		print "p_c:"
#		print p_c


	max_val = Z[-2,2]
	Z[-1] = [C[0],C[1],max_val+0.05,2]
#	Z.append([C[0],C[1],max_val+0.05,2])
#	print Z
#	print len(Z)
#	print X
#		print p_c_x
#	

	if visual ==1:
		plt.figure(figsize=(25, 10))
		dendrogram(Z)
		plt.show()

	return Z, C



def viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen=50,numMix=5):
	# Modify the speaker boundaries after 1st pass alignment
	# Inputs:
	# mfcc:	Frame-level features
	#	A numpy array of size [N_frames,d]
	# frameClust: Speaker hypothesis values at frame level 
	#	A numpy array of length N_frames
	# segLen: Length of segment (in seconds)
	#	A scalar
	# frameRate: Number of frames per seconds
	#	A scalar
	# minBlockLen: Minimum speaker segment length (in frames)
	#	A scalar
	# numMix: Number of Gaussian per speaker
	#	A scalar
	#
	# Outputs:
	# optimalStateSeq: Frame-level speaker hypothesis
	#	A numpy array of length N_frames
	#
	#
	# NOTE: This has been hard-coded for 2 speakers as of now

	
	# Define an ergodic HMM
	p12 = len(np.where(np.ediff1d(frameClust)==1)[0])
	p21 = len(np.where(np.ediff1d(frameClust)==-1)[0])
	p11,p22 = 0,0
	for x,y in zip(frameClust,frameClust[1:]):	
		if x==y:
			if x==1:
				p11 += 1
			else:
				p22 += 1
	p11,p12 = np.log(float(p11)/(p11+p12)), np.log(float(p12)/(p11+p12))
	p21,p22 = np.log(float(p21)/(p21+p22)), np.log(float(p22)/(p21+p22))

	# Train a Gaussian (maybe, GMM) using features for each speaker
	spkr1Feat = mfcc[frameClust==1,:]
	spkr2Feat = mfcc[frameClust==2,:]
	spkr1Gauss = GaussianMixture(n_components=numMix,covariance_type='diag').fit(spkr1Feat)
	spkr2Gauss = GaussianMixture(n_components=numMix,covariance_type='diag').fit(spkr2Feat)

	# Viterbi re-alignment
	print("Performing Viterbi realignment...")
	n_blocks = int(np.ceil(float(mfcc.shape[0])/minBlockLen))
	featBlocks = []
	for blockI in range(n_blocks-1):
		featBlocks += [mfcc[blockI*minBlockLen:(blockI+1)*minBlockLen,:]]
	featBlocks += [mfcc[(blockI+1)*minBlockLen:,:]]

	optimalScore = np.zeros((2,n_blocks))
	optimalPrevState = np.zeros((2,n_blocks))
	
		# Initializing for the first block
	optimalScore[0,0] = sum(spkr1Gauss.score_samples(featBlocks[0]))
	optimalScore[1,0] = sum(spkr2Gauss.score_samples(featBlocks[1]))
	optimalPrevState[0,0] = 0
	optimalPrevState[1,0] = 1	
	
		# Computing for intermediate blocks
	for blockI in range(1,n_blocks):
		temp1 = [optimalScore[0,blockI-1] + p11 + minBlockLen*p11 + sum(spkr1Gauss.score_samples(featBlocks[blockI])), optimalScore[1,blockI-1] + p21 + minBlockLen*p22 + sum(spkr1Gauss.score_samples(featBlocks[blockI]))]
		optimalPrevState[0,blockI] = np.argmax(temp1)
		optimalScore[0,blockI] = max(temp1)

		temp2 = [optimalScore[0,blockI-1] + p12 + minBlockLen*p11 + sum(spkr2Gauss.score_samples(featBlocks[blockI])), optimalScore[1,blockI-1] + p22 + minBlockLen*p22 + sum(spkr2Gauss.score_samples(featBlocks[blockI]))]
		optimalPrevState[1,blockI] = np.argmax(temp2)
		optimalScore[1,blockI] = max(temp2)
		
		# Backtracking
	optimalStateSeq = -1*np.ones(n_blocks)	
	optimalStateSeq[-1] = np.argmax(optimalScore[:,-1])
	for blockI in range(n_blocks-1,0,-1):
		optimalStateSeq[blockI-1] = optimalPrevState[int(optimalStateSeq[blockI]),blockI]
	optimalStateSeq = medfilt(optimalStateSeq,int(np.ceil(frameRate/minBlockLen) // 2 * 2 + 1))

	# Reconvert block to frame level scores
	frameClust = np.zeros(mfcc.shape[0])
	for blockI in range(n_blocks-1):
		frameClust[blockI*minBlockLen:(blockI+1)*minBlockLen] = optimalStateSeq[blockI]*np.ones(minBlockLen)
	frameClust[(blockI+1)*minBlockLen:] = optimalStateSeq[blockI]*np.ones(mfcc.shape[0]-(blockI+1)*minBlockLen)

	return 1+frameClust
