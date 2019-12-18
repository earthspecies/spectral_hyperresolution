# Spectral Hyperresolution

This repository brings [Sparse time-frequency representations](https://doi.org/10.1073/pnas.0601707103) by Timothy J. Gardner and Marcelo O. Magnasco to the Python world.

To quote the paper:

> Many details of complex sounds that are virtually undetectable in standard sonograms are readily perceptible and visible in reassignment

The idea we would like to explore is the application of the novel linear reassignment technique to downstream tasks (audio classification, unsupervised machine translation, etc). The hope is that this richer representation can lead to improved performance.

![Example of Hyperresolution Spectrogram](https://raw.githubusercontent.com/earthspecies/spectral_hyperresolution/master/data/dolphin_hyper.png)

### Installation

To install from github run `pip install git+git://github.com/earthspecies/spectral_hyperresolution.git`

### Table of Contents

1. [Introduction to linear reassignment in Python](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb) with extensive discussion of parameters.
2. Implementation of linear reassignment in [Pytorch](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_pytorch.py) and [Numpy](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment.py)
3. [Verification of correctness of implementations as well as comparison of their execution times](https://github.com/earthspecies/spectral_hyperresolution/blob/master/verify_correctness_and_benchmark.ipynb) using [data exported from MATLAB](https://github.com/earthspecies/spectral_hyperresolution/blob/master/save_MATLAB_data_for_verifying_correctness.ipynb).
4. [Introduction to linear reassignment in MATLAB](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_MATLAB.ipynb) with extensive discussion of parameters.
5. [Original MATLAB implementation](https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m).

### Useful resources

* [Instructions on configuring Jupyter Notebook to work with Matlab](https://am111.readthedocs.io/en/latest/jmatlab_install.html) - this is not required to run Python examples, but necessary if you would like to run the NB that uses the Matlab kernel
* [A Compact Primer On Digital Signal Processing](https://jackschaedler.github.io/circles-sines-signals/index.html) - if you are new to DSP, this is a very gentle introduction (with a lot of links to additional materials), great interactive visualizations, definitely worth checking out


### Special thanks

I would like to extend my gratitude to Marcelo O. Magnasco, the discoverer of linear reassignment. He has been extremely kind and patient to explain to me the behavior of linear reassingment and to share his code. His help is what made the work in this repository possible.

"Sparse Time-Frequency Representations",Timothy J. Gardner and Marcelo O. Magnasco Proc. Natl. Acad. Sci. USA103.16 6094-6099 (2006)
