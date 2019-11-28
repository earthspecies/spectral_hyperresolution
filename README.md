### !!! please note: this repository is a WIP !!!


## Spectral Hyperresolution

This repository brings [Sparse time-frequency representations](https://doi.org/10.1073/pnas.0601707103) by Timothy J. Gardner and Marcelo O. Magnasco to the Python world.

To quote the paper:

> Many details of complex sounds that are virtually undetectable in standard sonograms are readily perceptible and visible in reassignment

The idea we would like to explore is the application of the novel linear reassignment technique to downstream tasks (audio classification, unsupervised machine translation, etc). The hope is that this richer representation can lead to improved performance.

### Table of Contents

1. [Original Matlab implementation](https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m).
2. [Introduction to linear reassignment](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example.ipynb) with extensive discussion of parameter values.
3. Two NBs ([Matlab](https://github.com/earthspecies/spectral_hyperresolution/blob/master/save_data_to_help_with_Python_implementation.ipynb), [Python](https://github.com/earthspecies/spectral_hyperresolution/blob/master/implement_linear_reassignment_in_Python.ipynb)) used while porting the algorithm.
4. A Python module - [linear_reassignment.py](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment.py) - containing the implementation of linear reassignment.
5. A [notebook](https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_in_Python.ipynb) walking through the functionality imported from `linear_reassignment.py`.
6. [Visualizing synthetic sounds](https://github.com/earthspecies/spectral_hyperresolution/blob/master/visualizing_synthetic_sounds.ipynb) - experimenting with linear reassignment in a controlled environment.

### Useful resources

* [Instructions on configuring Jupyter Notebook to work with Matlab](https://am111.readthedocs.io/en/latest/jmatlab_install.html) - this is not required to run Python examples, but necessary if you would like to run the NB that uses the Matlab kernel
* [A Compact Primer On Digital Signal Processing](https://jackschaedler.github.io/circles-sines-signals/index.html) - if you are new to DSP, this is a very gentle introduction (with a lot of links to additional materials), great interactive visualizations, definitely worth checking out


### Special thanks

I would like to extend my gratitude to Marcelo O. Magnasco, the discoverer of linear reassignment. He has been extremely kind and patient to explain to me the behavior of linear reassingment and to share his code. His help is what made the work in this repository possible.

"Sparse Time-Frequency Representations",Timothy J. Gardner and Marcelo O. Magnasco Proc. Natl. Acad. Sci. USA103.16 6094-6099 (2006)
