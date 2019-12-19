# Spectral Hyperresolution

This repository brings [Sparse time-frequency representations](https://doi.org/10.1073/pnas.0601707103) by Timothy J. Gardner and Marcelo O. Magnasco to the Python world.

To quote the paper:

> Many details of complex sounds that are virtually undetectable in standard sonograms are readily perceptible and visible in reassignment

The idea we would like to explore is the application of the novel linear reassignment technique to downstream tasks (audio classification, unsupervised machine translation, etc). The hope is that this richer representation can lead to improved performance.

To observe the high resolution linear reassignment can achieve, please click on the image below and zoom in.
![Example of Hyperresolution Spectrogram](https://raw.githubusercontent.com/earthspecies/spectral_hyperresolution/master/data/dolphin_hyper.png)

### Installation

To install from github run `pip install git+git://github.com/earthspecies/spectral_hyperresolution.git`

### Table of Contents

1. [Introduction to linear reassignment](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_overview.ipynb) with extensive discussion of parameters.
2. Implementation of linear reassignment in [Pytorch](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_pytorch.py) and [Numpy](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment.py)
3. [Verification of correctness of implementations as well as comparison of their execution times](https://github.com/earthspecies/spectral_hyperresolution/blob/master/verify_correctness_and_benchmark.ipynb) using [data exported from MATLAB](https://github.com/earthspecies/spectral_hyperresolution/blob/master/save_MATLAB_data_for_verifying_correctness.ipynb).
4. [Original MATLAB implementation](https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m).

### Special thanks

I would like to extend my gratitude to Marcelo O. Magnasco, the discoverer of linear reassignment. He has been extremely kind and patient to explain to me the behavior of linear reassingment and to share his code. His help is what made the work in this repository possible.

"Sparse Time-Frequency Representations",Timothy J. Gardner and Marcelo O. Magnasco Proc. Natl. Acad. Sci. USA103.16 6094-6099 (2006)
