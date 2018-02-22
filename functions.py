#!/bin/python

# Date created: Nov 5 2017

# Defines various functions necessary for IB-based speaker clustering

import os
import pickle
import warnings
import librosa
import kaldi_io
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage.filters import *
from scipy.signal import *
from sklearn.mixture import *
from numpy.matlib import repmat
from collections import Counter

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

# NOTE: Both JS_div() and KL_div() functions are not used 'as-is' anymore. Refer to fastverbose_computeDeltaObj()
# Compute the Jenson-Shannon divergence for the two variables with distributions p_z_c(z|i) and p_z_c(z|j)
def JS_div(p_z_c,p_c,i,j,call):    
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
            mysum = 10,000
    return mysum


def fastverbose_oneBigFunction(p_y_c_i, p_y_c_j, p_x_c_i, p_x_c_j, side_p_y_c_i, side_p_y_c_j, weighted_p_y_c_i, weighted_p_y_c_j, p_c, i, j, beta, alpha, weights):

    # Computes the change in objective function resulting from merging clusters 'i' and 'j', given the 
    # conditional probabilities p(y|c), p(x|c) and the cluster weights p(c). Refer to JS_div() and KL_div() 
    # functions for a more detailed alternative
    #
    # Input:
    # p_y_c_i: p_y_c[i]; p_y_c_j: p_y_c[j]; 
    # p_x_c_i: p_x_c[i]; p_x_c_j: p_x_c[j]
    # side_p_y_c_i: side_p_y_c[i]; side_p_y_c_j: side_p_y_c[j]; 
    # weighted_p_x_c_i: weighted_p_x_c[i]; weighted_p_x_c_j: weighted_p_x_c[j]
    # alpha, beta: scalars for weighing the distributions p(y|c) and side_p(y|c) respectively
    # weights: vector for weighing weighted weighted_p(y|c). Same length as number of mixtures in weighted_p(y|c)
    #
    # Output: 
    #     (p_c[i] + p_c[j])*(JS_div(p(y|c),p_c,i,j) - (1/beta)*JS_div(p(x|c,)p_c,i,j)) - (1/alpha)*JS_div(side_p(y|c), p_c,i,j) - (1/weights)*JS_div(weighted_p(y|c),p_c,i,j)
    # 

    pie_i, pie_j = p_c[i]/(p_c[i] + p_c[j]), p_c[j]/(p_c[i] + p_c[j])
    if np.any(pie_i*p_y_c_i + pie_j*p_y_c_j==0) is True:
        return 10,000	# do not cluster
    if np.any(pie_i*p_x_c_i + pie_j*p_x_c_j==0) is True:
        return -10,000	# definitely cluster
    if side_p_y_c_i is not None and np.any(pie_i*side_p_y_c_i + pie_j*side_p_y_c_j==0) is True:
        return -10,000	# definitely cluster

    q_z = pie_i*p_y_c_i + pie_j*p_y_c_j
    nzIndices = np.where(p_y_c_i!=0)
    klterm1 = np.sum(np.multiply(p_y_c_i[nzIndices],np.log2(np.divide(p_y_c_i[nzIndices],q_z[nzIndices]))))
    nzIndices = np.where(p_y_c_j!=0)
    klterm2 = np.sum(np.multiply(p_y_c_j[nzIndices],np.log2(np.divide(p_y_c_j[nzIndices],q_z[nzIndices]))))
    term1 = pie_i*klterm1 + pie_j*klterm2

    if beta == 0:
	beta,term2 = 1,0
    else:
	q_z = pie_i*p_x_c_i + pie_j*p_x_c_j
	nzIndices = np.where(p_x_c_i!=0)
	klterm1 = np.sum(np.multiply(p_x_c_i[nzIndices],np.log2(np.divide(p_x_c_i[nzIndices],q_z[nzIndices]))))
	nzIndices = np.where(p_x_c_j!=0)
	klterm2 = np.sum(np.multiply(p_x_c_j[nzIndices],np.log2(np.divide(p_x_c_j[nzIndices],q_z[nzIndices]))))
	term2 = pie_i*klterm1 + pie_j*klterm2

    if alpha == 0 or alpha is None:
	alpha,term3 = 1,0
    else:
	q_z = pie_i*side_p_y_c_i + pie_j*side_p_y_c_j
	nzIndices = np.where(side_p_y_c_i!=0)
	klterm1 = np.sum(np.multiply(side_p_y_c_i[nzIndices],np.log2(np.divide(side_p_y_c_i[nzIndices],q_z[nzIndices]))))
	nzIndices = np.where(side_p_y_c_j!=0)
	klterm2 = np.sum(np.multiply(side_p_y_c_j[nzIndices],np.log2(np.divide(side_p_y_c_j[nzIndices],q_z[nzIndices]))))
	term3 = pie_i*klterm1 + pie_j*klterm2

    if weights is None:
	term4 = 0
    else:
        q_z = pie_i*weighted_p_y_c_i + pie_j*weighted_p_y_c_j
        nzIndices = np.where(weighted_p_y_c_i!=0)
        klterm1 = np.dot(weights[nzIndices],np.multiply(weighted_p_y_c_i[nzIndices],np.log2(np.divide(weighted_p_y_c_i[nzIndices],q_z[nzIndices]))))
        nzIndices = np.where(weighted_p_y_c_j!=0)
        klterm2 = np.dot(weights[nzIndices],np.multiply(weighted_p_y_c_j[nzIndices],np.log2(np.divide(weighted_p_y_c_j[nzIndices],q_z[nzIndices]))))
        term4 = pie_i*klterm1 + pie_j*klterm2

    return (p_c[i] + p_c[j])*(term1 - (1/beta)*term2 - (1/alpha)*term3 + term4)





def doVADWithKaldi(wavFile, frameRate, kaldiRoot):
    # Classifies frames into voiced/unvoiced, using Kaldi's pov feature
    # Inputs:
    # wavFile: Full path to wave file
    #    A string
    # frameRate: Number of frames per seconds
    #    A scalar
    # kaldiRoot: Full path to root directory of kaldi installation
    #     A string


    with open("temp.scp","w") as input_scp:
        input_scp.write("temp %s" % wavFile)
    os.system(kaldiRoot+'/src/featbin/compute-kaldi-pitch-feats --frame-shift='+str(1000/frameRate)+' --min-f0=40 --max-f0=600 scp:temp.scp ark:pitch.ark')
    for key,mat in kaldi_io.read_mat_ark('pitch.ark'):
        nccf = mat[:,0]
    l = -5.2 + 5.4*np.exp(7.5*(abs(nccf)-1)) + 4.8*abs(nccf) -2*np.exp(-10*abs(nccf)) + 4.2*np.exp(20*(abs(nccf)-1))
    p = gaussian_filter1d(1./(1 + np.exp(-l)),np.sqrt(10))
    vad = p>0.1
    os.system("rm temp.scp pitch.ark")
    return medfilt(vad,21).astype('bool')

def doVADWithSklearn(wavData, frameRate):
    # An alternative to doVADKaldi - uses the librosa library to compute short-term energy

    ste = librosa.feature.rmse(wavData,hop_length=int(16000/frameRate)).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    return (ste>thresh).astype('bool')


def trainGMMWithSklearn(wavFile, GMMfile, frameRate, segLen, vad, localGMM, numMix):
    # Given an audio file, train a GMM using the EM algorithm
    # Inputs:
    # wavFile: Full path to wave file
    #    A string
    # GMMfile: A pickle file with trained GMM model, if available
    #    A string
    # frameRate: Number of frames per seconds
    #    A scalar
    # segLen: Length of segment (in seconds)
    #    A scalar
    # vad:     Voiced activity decisions at frame level
    #    A numpy logical array
    # localGMM: Whether to disregard the model file and train a GMM locally 
    #     A boolean variable    
    # numMix: number of mixture in the GMM
    #    A scalar


    wavData,_ = librosa.load(wavFile,sr=16000)
    mfcc = librosa.feature.mfcc(wavData, sr=16000, n_mfcc=19,hop_length=int(16000/frameRate)).T
#    If using velocity & acceleration features
#    deltamfcc = librosa.feature.delta(mfcc.T,order=1).T
#    deltadeltamfcc = librosa.feature.delta(mfcc.T,order=2).T
#    mfcc = np.hstack((mfcc,deltamfcc,deltadeltamfcc))

    if vad is None:
        vad = doVADWithSklearn(wavData,frameRate)
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:]

    if localGMM == 1:
        print("Training GMM..")
        GMM = GaussianMixture(n_components=numMix,covariance_type='diag').fit(mfcc)
    else:
        print("Using available GMM model..")
        GMM = pickle.load(open(GMMfile,'rb'))

    var_floor = 1e-5
    segLikes = []
    segSize = frameRate*segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end of file
            break
        seg = mfcc[startI:endI,:]
        compLikes = np.sum(GMM.predict_proba(seg),0)
        segLikes.append(compLikes/seg.shape[0])

    return mfcc, vad, np.asarray(segLikes)


def trainGMMWithKaldi(wavFile, mdlFile, frameRate, segLen, kaldiRoot, vad, localGMM, numMix):
    # Given an audio file and GMM trained in Kaldi, compute mfcc features and frame-level posteriors
    # Inputs:
    # wavFile: Full path to wave file
    #    A string
    # mdlFile: Full path to Kaldi model file
    #     A string
    # frameRate: Number of frames per seconds
    #    A scalar
    # segLen: Length of segment (in seconds)
    #    A scalar
    # kaldiRoot: Full path to root directory of kaldi installation
    #     A string
    # vad:     Voiced activity decisions at frame level
    #    A numpy logical array
    # localGMM: Whether to disregard the model file and train a GMM locally 
    #     A boolean variable
    # numMix: number of mixture in the GMM. Relevant only if localGMM=True
    #    A scalar 


    os.system('mkdir local_kaldi_data')
    with open("local_kaldi_data/temp.scp","w") as input_scp:
        input_scp.write("temp %s" % wavFile)

    os.system(kaldiRoot+'/src/featbin/compute-mfcc-feats --frame-shift='+str(1000/frameRate)+' --frame-length=40 --use-energy=true --num-ceps=19 scp:local_kaldi_data/temp.scp ark:local_kaldi_data/raw.ark')
#    If using velocity & acceleration features
#    os.system(kaldiRoot+'/src/featbin/compute-mfcc-feats --frame-shift='+str(1000/frameRate)+' --frame-length=40 --use-energy=false --num-ceps=19 scp:local_kaldi_data/temp.scp ark:- | '+kaldiRoot+'/src/featbin/add-deltas ark:- ark:local_kaldi_data/raw.ark')
    os.system(kaldiRoot+'/src/featbin/compute-cmvn-stats ark:local_kaldi_data/raw.ark ark:local_kaldi_data/cmvn.ark')
    os.system(kaldiRoot+'/src/featbin/apply-cmvn ark:local_kaldi_data/cmvn.ark ark:local_kaldi_data/raw.ark ark,scp:local_kaldi_data/out.ark,local_kaldi_data/out.scp')
    for key,mat in kaldi_io.read_mat_ark('local_kaldi_data/out.ark'):
        if vad is None:
            vad = doVADWithKaldi(wavFile,frameRate,kaldiRoot)	
        if mat.shape[0] > vad.shape[0]:
            vad = np.hstack((vad,np.zeros(mat.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
        elif mat.shape[0] < vad.shape[0]:
            vad = vad[:mat.shape[0]]
        mfcc = mat[vad,:]

    if localGMM == False:
        numMix = os.popen(kaldiRoot+'/src/gmmbin/gmm-global-info '+mdlFile+' | grep "number of gaussians" | awk \'{print $NF}\'').readlines()[0].strip('\n')
        os.system(kaldiRoot+'/src/gmmbin/gmm-global-get-post --n='+numMix+' '+mdlFile+' ark:local_kaldi_data/out.ark ark:local_kaldi_data/post.ark')
    else:
        pwd = os.getcwd()
        os.system("sed \"s~local_kaldi_data~${PWD}/local_kaldi_data~g\" local_kaldi_data/out.scp > local_kaldi_data/feats.scp")
        os.system("echo \"temp temp\" > local_kaldi_data/utt2spk")
        os.system("bash train_diag_ubm.sh --num-iters 20 --num-frames 500000 --nj 1 --num-gselect "+str(numMix)+" "+pwd+"/local_kaldi_data/ "+str(numMix)+" "+pwd+"/local_kaldi_data/")
        os.system(kaldiRoot+'/src/gmmbin/gmm-global-get-post --n='+str(numMix)+' local_kaldi_data/final.dubm ark:local_kaldi_data/out.ark ark:local_kaldi_data/post.ark')
    
    for key,post in kaldi_io.read_post_ark('local_kaldi_data/post.ark'):
        # Sort posteriors according to the mixture index
        for frameI in range(len(post)):
            post[frameI] = sorted(post[frameI],key=lambda x: x[0])
    post = np.asarray(post)[:,:,1]
    post = post[vad]  

    segSize = frameRate*segLen
    segLikes = []
    for segI in range(int(np.ceil(float(post.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > post.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end 
            break        
        segLikes.append(np.mean(post[startI:endI,:],axis=0))

    os.system("rm -rf local_kaldi_data")
    return mfcc, vad, np.asarray(segLikes)


def convertDecisionsSegToFrame(clust, segLen, frameRate, numFrames):
    # Convert cluster assignemnts from segment-level to frame-level
    # Inputs:
    # clust: Speaker hypothesis values at segment level 
    #    A numpy array of length N 
    # segLen: Length of segment (in seconds)
    #    A scalar
    # frameRate: Number of frames per seconds
    #    A scalar
    # numFrames: total number of voiced frames


    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust)-1):
        frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
    frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI+1]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
    return frameClust


def getSidePosteriors(wavFile, kaldiRoot, frameRate, segLen, vad, pbm, numFrames):
    os.system('mkdir local_kaldi_data')
    with open("local_kaldi_data/temp.scp","w") as input_scp:
        input_scp.write("temp %s" % wavFile)

    os.system(kaldiRoot+'/src/featbin/compute-mfcc-feats --frame-shift='+str(1000/frameRate)+' --frame-length=40 --use-energy=true --num-ceps=19 scp:local_kaldi_data/temp.scp ark:local_kaldi_data/raw.ark')
    os.system(kaldiRoot+'/src/featbin/compute-cmvn-stats ark:local_kaldi_data/raw.ark ark:local_kaldi_data/cmvn.ark')
    os.system(kaldiRoot+'/src/featbin/apply-cmvn ark:local_kaldi_data/cmvn.ark ark:local_kaldi_data/raw.ark ark,scp:local_kaldi_data/out.ark,local_kaldi_data/out.scp')
    numMix = os.popen(kaldiRoot+'/src/gmmbin/gmm-global-info '+pbm+' | grep "number of gaussians" | awk \'{print $NF}\'').readlines()[0].strip('\n')
    os.system(kaldiRoot+'/src/gmmbin/gmm-global-get-post --n='+str(numMix)+' '+pbm+' ark:local_kaldi_data/out.ark ark:local_kaldi_data/post.ark')
    for key,post in kaldi_io.read_post_ark('local_kaldi_data/post.ark'):
        # Sort posteriors according to the mixture index
        for frameI in range(len(post)):
            post[frameI] = sorted(post[frameI],key=lambda x: x[0])
    post = np.asarray(post)[:,:,1]
    post = post[vad]  

    segSize = frameRate*segLen
    segLikes = []
    for segI in range(int(np.ceil(float(post.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > post.shape[0]:
            endI = numFrames-1
        if endI==startI:    # Reached the end 
            break        
        segLikes.append(np.mean(post[startI:endI,:],axis=0))

    os.system("rm -rf local_kaldi_data")
    return np.asarray(segLikes)



def oneBigFunction(p_y_x, side_p_y_x, weighted_p_y_x, beta, alpha, weights, visual):
    # The main clustering function - performs bottom-up clustering using the IB criterion
    # Inputs:
    # p_y_x: Conditional probability p(y|x)
    #    A numpy array of size [N,P]
    # beta: Tradeoff parameter in the IB objective
    #     A scalar
    # visual: Print dendrogram
    #    Boolean value
    #
    # Outputs:
    # C: Cluster assignment; an m-partitiion of X, 1 <= m <= |X| 
    #    A numpy array of size [N,1]
    #
    # Objective: Min (1/beta)*I(X,C) - I(Y,C) + (1/alpha)*I(Y',C) + weights*I(Y_weighted,C)
    # X: Features at segment-level
    # Y: Relevance variable, typically components from a GMM
    # Y': 'Irrelevant' variable, also components form a GMM
    # Y_weighted: Generalization of Y, GMM with component-specific weights specified using the vector 'weights'
    #
    # NOTE: This function ALWAYS creates 2 clusters. Use the fcluster() method to prune the dendrogram 
    # variable with the desired criterion. Refer infoBottleneck.py for usage

    np.random.seed(1000)
    print("Performing agglomerative clustering using IB objective...")
    N,P = np.shape(p_y_x)
    p_c = np.empty(N)
    p_y_c = np.empty((N,P))    # p(y|c), NOT p(y,c)

    if side_p_y_x is not None:
	Q = np.shape(side_p_y_x)[1]
	side_p_y_c = np.empty((N,Q))
    if weighted_p_y_x is not None:
	R = np.shape(weighted_p_y_x)[1]
        weighted_p_y_c = np.empty((N,R))	

    p_c_x = np.zeros((N,N))
    p_x_c = np.zeros((N,N))
    delta_F = np.zeros((N,N))
    N_init = N

    print("Initialization...")
    C = range(N)
    for i in range(N):
        p_c[i] = 1.0/N
        p_c_x[i,i] = 1.0
        p_x_c[i,i] = 1.0
        for j in range(P):
            p_y_c[i,j] = p_y_x[i,j]
	if side_p_y_x is not None:
            for j in range(Q):
	        side_p_y_c[i,j] = side_p_y_x[i,j]
	if weighted_p_y_x is not None:
            for j in range(R):
	        weighted_p_y_c[i,j] = weighted_p_y_x[i,j]



    if side_p_y_x is not None and weighted_p_y_x is not None:

        for i in range(N):
            for j in range(i):
                delta_F[i,j] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[j,:], p_x_c[i,:], p_x_c[j,:], side_p_y_c[i,:], side_p_y_c[j,:], weighted_p_y_c[i,:], weighted_p_y_c[j,:], p_c, i, j, beta, alpha, weights)
            for j in range(i,N):
                delta_F[i,j] = float("inf")

    elif side_p_y_x is not None:

        for i in range(N):
            for j in range(i):
                delta_F[i,j] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[j,:], p_x_c[i,:], p_x_c[j,:], side_p_y_c[i,:], side_p_y_c[j,:], None, None, p_c, i, j, beta, alpha, None)
            for j in range(i,N):
                delta_F[i,j] = float("inf")

    elif weighted_p_y_x is not None:

        for i in range(N):
            for j in range(i):
                delta_F[i,j] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[j,:], p_x_c[i,:], p_x_c[j,:], None, None, weighted_p_y_c[i,:], weighted_p_y_c[j,:], p_c, i, j, beta, None, weights)
            for j in range(i,N):
                delta_F[i,j] = float("inf")

    else:

        for i in range(N):
            for j in range(i):
                delta_F[i,j] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[j,:], p_x_c[i,:], p_x_c[j,:], None, None, None, None, p_c, i, j, beta, None, None)
            for j in range(i,N):
                delta_F[i,j] = float("inf")


    # Clustering
    max_clust_ind = max(C)
    Z = np.empty((max_clust_ind,4))
    curr_val = 0
    iterIndex = 0
    print("Number of clusters = "+str(N))

    while len(np.unique(C))>2:
        if N%100==0:
            print("Number of clusters = "+str(N))

#        print("Performing one iteration of clustering..")
        [i_opt,j_opt] = np.unravel_index(np.argmin(delta_F), delta_F.shape)
#        print ("Optimal indices: ("+str(i_opt)+","+str(j_opt)+")")
        curr_val += abs(np.min(delta_F))
        Z[iterIndex] = [C[i_opt],C[j_opt],curr_val,2]
        iterIndex += 1

        # Create temporary variables for storing the new distributions
        C_new = []
        p_c_new = []
        for i in range(N):
            if i!=i_opt and i!=j_opt:
                C_new.append(C[i])
                p_c_new.append(p_c[i])

        p_y_c_new = np.delete(p_y_c,(i_opt,j_opt),0)
	if side_p_y_x is not None:
	    side_p_y_c_new = np.delete(side_p_y_c,(i_opt,j_opt),0)
	if weighted_p_y_x is not None:
	    weighted_p_y_c_new = np.delete(weighted_p_y_c,(i_opt,j_opt),0)
        p_c_x_new = np.delete(p_c_x,(i_opt,j_opt),1)
        delta_F = np.delete(np.delete(delta_F,(i_opt,j_opt),0),(i_opt,j_opt),1)

        # Update p(y|c)
        C_new.append(max_clust_ind+1)
        temp1 = np.zeros(P)
        for j in range(P):
            temp1[j] = (p_y_c[i_opt,j]*p_c[i_opt] + p_y_c[j_opt,j]*p_c[j_opt])/(p_c[i_opt] + p_c[j_opt])
        p_y_c_new = np.vstack((p_y_c_new,temp1))

	if side_p_y_x is not None:
            temp2 = np.zeros(Q)
	    for j in range(Q):
                temp2[j] = (side_p_y_c[i_opt,j]*p_c[i_opt] + side_p_y_c[j_opt,j]*p_c[j_opt])/(p_c[i_opt] + p_c[j_opt])
            side_p_y_c_new = np.vstack((side_p_y_c_new,temp2))

	if weighted_p_y_x is not None:
            temp3 = np.zeros(R)
	    for j in range(R):
	        temp3[j] = (weighted_p_y_c[i_opt,j]*p_c[i_opt] + weighted_p_y_c[j_opt,j]*p_c[j_opt])/(p_c[i_opt] + p_c[j_opt])
	    weighted_p_y_c_new = np.vstack((weighted_p_y_c_new,temp3))

        # Update p(c|x)
        temp = np.zeros(N_init)
        for i in range(N):
            if i!=i_opt and i!=j_opt:
                temp[i] = 0
            else:
                temp[i] = 1
        p_c_x_new = np.concatenate((p_c_x_new,np.reshape(temp,(len(temp),1))),1)
    
        # Update p(c)
        p_c_new.append(p_c[i_opt] + p_c[j_opt])
        max_clust_ind += 1
        C = C_new
        p_y_c = p_y_c_new
        p_y_c[p_y_c<10e-10] = 0.
	if side_p_y_x is not None:
            side_p_y_c = side_p_y_c_new
	    side_p_y_c[side_p_y_c<10e-10] = 0.
	if weighted_p_y_x is not None:
	    weighted_p_y_c = weighted_p_y_c_new
            weighted_p_y_c[weighted_p_y_c<10e-10] = 0.
        p_c_x = p_c_x_new
        p_c = np.asarray(p_c_new)

        # Update p(x|c)
        p_x_c = np.divide(p_c_x.T,N_init*repmat(p_c,N_init,1).T) # this should be of shape (N-1,N_init)

        N -= 1
        p_c_x[p_c_x<10e-10] = 0.
        p_x_c[p_x_c<10e-10] = 0.
        p_c[p_c<10e-10] = 0.


        # Update delta_F
        # Add a row 
        newrow = np.zeros(N-1)
        for i in range(N-1):
	    if side_p_y_x is not None and weighted_p_y_x is not None:
                newrow[i] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[len(p_c)-1,:], p_x_c[i,:], p_x_c[len(p_c)-1,:], side_p_y_c[i,:], side_p_y_c[len(p_c)-1,:], weighted_p_y_c[i,:], weighted_p_y_c[len(p_c)-1,:], p_c, i, len(p_c)-1, beta, alpha, weights)
	    elif side_p_y_x is not None:
                newrow[i] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[len(p_c)-1,:], p_x_c[i,:], p_x_c[len(p_c)-1,:], side_p_y_c[i,:], side_p_y_c[len(p_c)-1,:], None, None, p_c, i, len(p_c)-1, beta, alpha, None)
	    elif weighted_p_y_x is not None:
                newrow[i] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[len(p_c)-1,:], p_x_c[i,:], p_x_c[len(p_c)-1,:], None, None, weighted_p_y_c[i,:], weighted_p_y_c[len(p_c)-1,:], p_c, i, len(p_c)-1, beta, None, weights)
	    else:
                newrow[i] = fastverbose_oneBigFunction(p_y_c[i,:], p_y_c[len(p_c)-1,:], p_x_c[i,:], p_x_c[len(p_c)-1,:], None, None, None, None, p_c, i, len(p_c)-1, None, None, None)
	
        # Add a column of "inf"
        newcol = float("inf")*np.ones(N)

        delta_F = np.concatenate((np.vstack((delta_F,newrow)),np.reshape(newcol,(len(newcol),1))),1)


    # Complete the dendrogram variable
    max_val = Z[-2,2]
    Z[-1] = [C[0],C[1],max_val+0.01,2]

    # Visualization, not really feasible for large utterances
    if visual ==1:
        plt.figure(figsize=(25, 10))
        dendrogram(Z)
        plt.show()

    return Z, C




def viterbiRealignment(mfcc,frameClust,segLen,frameRate,minBlockLen,numMix=5):
    # Modify the speaker boundaries after 1st pass alignment
    # Inputs:
    # mfcc:    Frame-level features
    #    A numpy array of size [N_frames,d]
    # frameClust: Speaker hypothesis values at frame level 
    #    A numpy array of length N_frames
    # segLen: Length of segment (in seconds)
    #    A scalar
    # frameRate: Number of frames per seconds
    #    A scalar
    # minBlockLen: Minimum speaker segment length (in frames)
    #    A scalar
    # numMix: Number of Gaussian components per speaker
    #    A scalar
    #
    # Outputs:
    # optimalStateSeq: Frame-level speaker hypothesis
    #    A numpy array of length N_frames


    # Define an ergodic HMM
    eps = 10e-10
    numSpkrs = len(np.unique(frameClust))
    bigramCounts = Counter(zip(frameClust.astype('int'), frameClust[1:].astype('int')))    
    p = np.empty((numSpkrs,numSpkrs))
    for i in range(numSpkrs):
        for j in range(numSpkrs):
            p[i,j] = 1.0*bigramCounts[(i+1,j+1)]
    p[p<eps] = eps
    p = np.log(np.divide(p,repmat(np.sum(p,1),numSpkrs,1).T))

    # Train Gaussians for each speaker using the current segmentation
    spkrFeats = []
    spkrGauss = []
    for spkrI in range(numSpkrs):
        spkrFeats.append(mfcc[frameClust==spkrI+1,:])
        try:
            spkrGauss.append(GaussianMixture(n_components=numMix,covariance_type='diag').fit(spkrFeats[-1]))
        except ValueError:
            spkrGauss.append(GaussianMixture(n_components=1,covariance_type='diag').fit(spkrFeats[-1]))


    # Viterbi re-alignment
    print("Performing Viterbi realignment...")
    n_blocks = int(np.ceil(float(mfcc.shape[0])/minBlockLen))
    featBlocks = []
    for blockI in range(n_blocks-1):
        featBlocks += [mfcc[blockI*minBlockLen:(blockI+1)*minBlockLen,:]]
    featBlocks += [mfcc[(blockI+1)*minBlockLen:,:]]

    optimalScore = np.zeros((numSpkrs,n_blocks))
    optimalPrevState = np.zeros((numSpkrs,n_blocks))
    
    # Initializing for the first block
    for spkrI in range(numSpkrs):
        optimalScore[spkrI,0] = sum(spkrGauss[spkrI].score_samples(featBlocks[0]))
        optimalPrevState[spkrI,0] = spkrI

    # Computing for intermediate blocks
    for blockI in range(1,n_blocks):
        for targetSpkrI in range(numSpkrs):
            temp = []
            for sourceSpkrI in range(numSpkrs):
                temp.append(float(optimalScore[sourceSpkrI,blockI-1] + p[sourceSpkrI,targetSpkrI] + minBlockLen*p[sourceSpkrI,sourceSpkrI] + sum(spkrGauss[targetSpkrI].score_samples(featBlocks[blockI]))))
            optimalPrevState[targetSpkrI,blockI] = np.argmax(temp)
            optimalScore[targetSpkrI,blockI] = max(temp)        


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

def writeRttmFile(pass4hyp, frameRate, wavFile, rttmFile):
    # Takes diarization results and creates a file in the RTTM format
    # pass4hyp: Hypothesis at frame-level. '-1' for unvoiced frames, 1,2,..N for speakers
    #     A numpy array of length N_frames
    # frameRate: Number of frames per seconds
    #    A scalar
    # wavFile: Full path to wave file
    #    A string
    # rttmFile: Full path to output RTTM file
    #    A string

    spkrChangePoints = np.where(pass4hyp[:-1] != pass4hyp[1:])[0]
    if spkrChangePoints[0]!=0 and pass4hyp[0]!=-1:
        spkrChangePoints = np.concatenate(([0],spkrChangePoints))
    spkrLabels = []    
    for spkrHomoSegI in range(len(spkrChangePoints)):
        spkrLabels.append(pass4hyp[spkrChangePoints[spkrHomoSegI]+1])

    fid = open(rttmFile,'w')
    for spkrI,spkr in enumerate(spkrLabels[:-1]):
        if spkr!=-1:
            fid.write("SPEAKER %s 0 %3.2f %3.2f <NA> <NA> spkr%d <NA>\n" %(wavFile.split('/')[-1].split('.')[0], (spkrChangePoints[spkrI]+1)/float(frameRate), (spkrChangePoints[spkrI+1]-spkrChangePoints[spkrI])/float(frameRate), spkr) )
    
    if spkrLabels[-1]!=-1:
        fid.write("SPEAKER %s 0 %3.2f %3.2f <NA> <NA> spkr%d <NA>\n" %(wavFile.split('/')[-1].split('.')[0], spkrChangePoints[-1]/float(frameRate), (len(pass4hyp) - spkrChangePoints[-1])/float(frameRate), spkrLabels[-1]) )
    fid.close()       
