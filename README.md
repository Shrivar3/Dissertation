# Dissertation: Uncertainty Quantification in Nested Sampling

Masters Dissertation Code – University of Warwick – 2026

This repository contains the code, experiments, and supporting material for my MSc dissertation at the University of Warwick:

**"Uncertainty Quantification in Nested Sampling"**

The project investigates bootstrap-based methods for estimating the uncertainty and bias of Nested Sampling evidence estimates, with a particular focus on stochastic shrinkage and phantom-point recycling.

---

## Author

**Shrivar Singh**  
MMORSE (Data Analysis Stream)  
University of Warwick  

Supervisors: Prof. Christian Robert & Dr. Nicholas Tawn

---

## Overview

Nested Sampling (Skilling, 2006) is a Monte Carlo method used to estimate the marginal likelihood: $\log Z = \log \int L(\theta)\,\pi(\theta)\,d\theta$

This quantity is central to Bayesian model comparison.

However, uncertainty quantification remains challenging due to:

- Stochastic shrinkage of prior volume  
- Dependence between samples  
- Constrained MCMC replacement steps  

This project develops and evaluates:

- Bootstrap Method (3): stochastic shrinkage ladder bootstrap  
- Phantom-point recycling  
- Bias correction via pooled and multi-run estimators  
- High-dimensional Nested Sampling implementations  

Bootstrap uncertainty estimates are compared against ground-truth multi-run Monte Carlo reference distributions.

---

## Repository Structure

SRC/  
Core implementation modules:

- run_ns_mh_phantom.py  
  Canonical Nested Sampling implementation with phantom-point storage

- bootstrap_method3.py  
  Bootstrap uncertainty estimator (Method 3)

- bootstrap_common.py  
  Shared utilities for bootstrap and evidence computation

- multi_run.py  
  Multi-run Nested Sampling reference generator


results/  

Saved experiment outputs:

- Multi-run reference distributions  
- Bootstrap outputs  
- Evidence (logZ) estimates  
- Saved ns_out objects  


notebooks/  

Jupyter notebooks used for:

- Analysis  
- QQ plots  
- Histogram comparisons  
- Bias and convergence diagnostics  


figures/  

Saved plots used in the dissertation  


dissertation/  

LaTeX source files for the dissertation  


---

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git  
cd YOUR_REPO  


Install the package in editable mode:

pip install -e .  


---

## Running Nested Sampling

Example usage:

from nsdissert import run_ns_mh_phantom  

ns_out = run_ns_mh_phantom(  
    n=600,  
    p=12,  
    n_live=400,  
    seed=123  
)  


---

## Running Bootstrap Method (3)

Example usage:

from nsdissert.bootstrap_method3 import method3_stochshrink_two_sided_chain  

results = method3_stochshrink_two_sided_chain(  
    ns_out,  
    n_bootstrap=500  
)  


---

## Reproducing Dissertation Results

To reproduce the main results:

1. Generate a multi-run Nested Sampling reference distribution  

2. Save ns_out objects  

3. Run Bootstrap Method (3)  

4. Compare bootstrap and reference distributions using:

   - QQ plots  
   - Histogram overlays  
   - Bias analysis  


All experiments are reproducible using the provided notebooks.


---

## Key Features

- Deterministic shrinkage Nested Sampling  
- Phantom-point storage and recycling  
- Bootstrap-based uncertainty estimation  
- Bias correction via pooled estimators  
- High-dimensional capability (tested for p > 1000)  
- Fully reproducible experimental pipeline  


---

## Dissertation Focus

Primary contribution:

Evaluation of bootstrap-based uncertainty quantification methods for Nested Sampling, including analysis of bias, variance, and convergence behaviour.


---

## References

Skilling, J. (2006). Nested Sampling for Bayesian Computations.


---

## License

Academic use only.


---

## Contact

Shrivar Singh  
University of Warwick  
