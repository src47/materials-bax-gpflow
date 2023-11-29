This repo contains the config files and GPflow code which accompanies the manuscript: "Targeted materials discovery using Bayesian Algorithm Execution". 

#### We recommend for practical usage of BAX methods to use the following repo which provides tutorials and a cleaner interface: https://github.com/src47/multibax-sklearn

## Installation via pip

To install, run:
```bash
$ pip install -r requirements.txt
```

## Running Examples

First make sure this repo directory is on the PYTHONPATH, e.g. by running:

```bash
$ export PYTHONPATH=$(pwd)    
```

And then in the main directory, run (changing the config file as desired): 

```bash
$ ./parallel.sh
```
# References 

For ternary phase diagram dataset [1,2] used in this work, please use the following instructions:
- Download the data file "FeCoNi_benchmark_dataset_220501a.mat" from https://github.com/usnistgov/remi/tree/nist-pages/data/Combinatorial%20Libraries/Fe-Co-Ni, convert to csv (named "ternary.csv") and place in the datasets directory. 

[1] Yoo, Young-kook et al. “Identification of amorphous phases in the Fe–Ni–Co ternary alloy system using continuous phase diagram material chips.” Intermetallics 14 (2006): 241-247.

[2] Alex Wang, Haotong Liang, Austin McDannald, Ichiro Takeuchi, Aaron Gilad Kusne, Benchmarking active learning strategies for materials optimization and discovery, Oxford Open Materials Science, Volume 2, Issue 1, 2022, itac006, https://doi.org/10.1093/oxfmat/itac006.

The nanoparticle synthesis data was obtained from an empirical model provided in [3].

[3] Pellegrino, Francesco, et al. "Machine learning approach for elucidating and predicting the role of synthesis parameters on the shape and size of TiO2 nanoparticles." Scientific Reports 10.1 (2020): 18910.

Code and methodology in this repo builds on InfoBAX [4] and Multi-point BAX [5]

[4] Neiswanger, Willie, Ke Alexander Wang, and Stefano Ermon. "Bayesian algorithm execution: Estimating computable properties of black-box functions using mutual information." International Conference on Machine Learning. PMLR, 2021.

[5] Miskovich, Sara A., et al. "Bayesian algorithm execution for tuning particle accelerator emittance with partial measurements." arXiv preprint arXiv:2209.04587 (2022).

# Contact information

Feel free to contact chitturi@stanford.edu, akashr@stanford.edu and willie.neiswanger@gmail.com for any questions regarding the code!
