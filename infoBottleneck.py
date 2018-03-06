#!/bin/python

# Date created: Nov 5 2017

#PRINT AN USAGE INFORMATION

import sys
import argparse
import numpy as np
from functions import *
from scipy.cluster.hierarchy import fcluster


parser = argparse.ArgumentParser()
parser.add_argument("--beta", help="Lagrangian parameter in the IB criterion (10)", default=10,type=float)
parser.add_argument("--segLen", help="Size of each segment during uniform segmentation (seconds, 2)",default=2,type=int)
parser.add_argument("--frameRate", help="Number of frames per second (Hz, 50). Must match with VAD file, if supplied",default=50,type=int)
parser.add_argument("--numCluster", help="Number of speakers (2)",default=2,type=int)
parser.add_argument("--library", help="For feature extractio and GMM training: kaldi/sklearn (kaldi)",default="kaldi")
parser.add_argument("--vadFile", help="Voiced(1)/Unvoiced(0) values. One frame per line",default=None)
parser.add_argument("--gmmFile", help="Pre-trained GMM model",default=None)
parser.add_argument("--localGMM", help="Whether to train a GMM locally (1) or not (0). This argument overrides --gmmFile",default=1,type=int)
parser.add_argument("--kaldiRoot", help="Kaldi root location",default=None)
parser.add_argument("--numMix", help="Number of Gaussian components. Will be over-written if supplied with pre-trained GMM model",default=64,type=int)
parser.add_argument("--minBlockLen", help="Minimum number of frames to be treated as a contiguous unit during re-alignment (Number of frames,25)",default=25,type=int)
parser.add_argument("--numRealignments", help="Number of Viterbi realignments (1)",default=1,type=int)
parser.add_argument("wavFile")
parser.add_argument("rttmFile")
args = parser.parse_args()


if args.localGMM == 0 and args.gmmFile == None:
    print("Please provide a GMM file if not training locally\n\n")
    parser.print_help()
    sys.exit(1)

if args.localGMM == 1 and args.gmmFile != None:
    print("Overriding pre-trained GMM model since localGMM is set to 1")

if args.library == "kaldi" and args.kaldiRoot == None:
    print("Please provide the kaldi root directory\n\n")
    parser.print_help()
    sys.exit(1)


np.random.seed(1000)
if args.vadFile is not None:
    vad = np.loadtxt(args.vadFile).astype('bool')
    #vad = np.interp(np.linspace(0,len(vad),int(len(vad)*args.frameRate/100.0)),np.arange(len(vad)),vad).astype('bool')
else:
    vad = None

if args.library == "kaldi":
    mfcc,vad,p_y_x = trainGMMWithKaldi(args.wavFile, args.gmmFile, args.frameRate, args.segLen, args.kaldiRoot, vad, args.localGMM, args.numMix)
else:
    mfcc,vad,p_y_x = trainGMMWithSklearn(args.wavFile, args.gmmFile, args.frameRate, args.segLen, vad, args.localGMM, args.numMix)

p_y_x[p_y_x<1e-10] = 1e-10
Z,C = cluster(p_y_x,args.beta,0)

clust = fcluster(Z,args.numCluster,criterion='maxclust')
frameClust = convertDecisionsSegToFrame(clust, args.segLen, args.frameRate, mfcc.shape[0])
pass1hyp = -1*np.ones(len(vad))
pass1hyp[vad] = frameClust

prevPassHyp = pass1hyp
for realignIter in range(args.numRealignments):
    frameClust = viterbiRealignment(mfcc,frameClust,args.segLen,args.frameRate,args.minBlockLen,numMix=16)
    nextPassHyp = -1*np.ones(len(vad))
    nextPassHyp[vad] = frameClust

    # If any speaker was lost during realignment, use hypothesis from previous iteration
    if len(np.unique(nextPassHyp)) < args.numCluster+1: 
        writeRttmFile(prevPassHyp, args.frameRate, args.wavFile, args.rttmFile)
        break
    else:
	prevPassHyp = nextPassHyp

writeRttmFile(nextPassHyp, args.frameRate, args.wavFile, args.rttmFile)
