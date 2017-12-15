# SAIL IB-Diarization Toolkit

The toolkit performs speaker diarization (finding 'who spoke when?') using the information bottleneck criterion. Specifically, it tries to group speech segments (X) into clusters \(C\) by minimizing the mutual information between them, while maximizing the mutual information between the segments and a set of relevance variables (Y). In the case of speaker diarization, the relevance variables are typically components of a GMM trained using all the voiced frames.

### Prerequisites

Python libraries: numpy, scipy, scikit-learn, librosa, [kaldi_io](https://github.com/vesis84/kaldi-io-for-python) (optional)
An installed Kaldi version is strongly recommended, but not mandatory

## Getting Started

Two excerpts from the AMI Meeting corpus are included alongwith manual annotations for speaker turns, labels and vad. Each file contains two speakers. Run the toolkit as follows:

```
python infoBottleneck.py ami_test_file/ES2002_test.wav ami_test_file/ES2002_test.vad ES2002.rttm 2
```

## Benchmarks
All values in Speaker Error (%)
| Method | AMI (ihm)  | ICSI |
| -------------| ------------- | ------------- |
| Bayesian Information Criterion  | 32.64  | 41.54 |
| [Idiap IB](https://github.com/idiap/IBDiarization) Toolkit | 27.55 | 38.35 |
| SAIL IB Toolkit | 28.40  | 39.50 |

## Authors

 Manoj Kumar (prabakar@usc.edu)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
