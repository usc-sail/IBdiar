# SAIL IB-Diarization Toolkit

The toolkit performs speaker diarization (finding 'who spoke when?') using the information bottleneck criterion. Specifically, it tries to group speech segments (X) into clusters \(C\) by minimizing the mutual information between them, while maximizing the mutual information between the segments and a set of relevance variables (Y). In the case of speaker diarization, the relevance variables are typically components of a GMM trained using all the voiced frames.

### Prerequisites

Python libraries: numpy, scipy, scikit-learn, librosa, [kaldi_io](https://github.com/vesis84/kaldi-io-for-python) (optional)  
An installed [Kaldi](https://github.com/kaldi-asr/kaldi) toolkit is highly recommended, but not mandatory

## Getting Started

For a quick demo, execute `runAMIExample.py` or `runSyntheticExample.py` without any arguments.   
The excerpts from AMI Meeting corpus come alongwith manual annotations for speaker turns, labels and vad. Each audio file contains two speakers. The synthetic example provides visualization using a dendrogram.

For a more comprehensive usage, refer to `infoBottleneck.py`
```
usage: infoBottleneck.py [-h] [--beta BETA] [--segLen SEGLEN] 
                         [--frameRate FRAMERATE] [--numCluster NUMCLUSTER]
                         [--library LIBRARY] [--vadFile VADFILE]
                         [--gmmFile GMMFILE] [--localGMM LOCALGMM]
                         [--kaldiRoot KALDIROOT] [--numMix NUMMIX]
                         [--minBlockLen MINBLOCKLEN]
                         [--numRealignments NUMREALIGNMENTS]
                         wavFile rttmFile
```
Execute with the help option for more information about each parameter, including default values. 

## Benchmarks
All values in Diarization Error Rate (%)

| Method | AMI (ihm)  | ICSI |
| -------------| ------------- | ------------- |
| Bayesian Information Criterion  | 32.64  | 41.54 |
| [Idiap IB](https://github.com/idiap/IBDiarization) Toolkit | 27.55 | 38.35 |
| SAIL IB Toolkit | 28.40  | 39.50 |

## Reference

D. Vijayasenan, F. Valente and H. Bourlard, "An Information Theoretic Approach to Speaker Diarization of Meeting Data," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 17, no. 7, pp. 1382-1393, Sept. 2009.

## Authors

 Manoj Kumar (prabakar@usc.edu)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
