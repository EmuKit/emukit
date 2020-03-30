This folder contains the source code for implementing the PBBO method discussed in the paper https://arxiv.org/abs/2003.11435

Start from installing the environment as instructed in installation.txt

After having installed the environment, the easiest way to understand how the code is ran 
(and to implement your own applications) is by following the example in
minimum_working_example.py.

If you want to gain deeper knowledge, the actual implementation can be found from the pbbo
folder and the structure of the code is as follows:

- bayesian_optimization.py implements the Bayesian optimization loop needed for 
  the PBBO algorithm.

- gp_models.py implements different GP models for all inference methods discussed in
  the paper. The actual inference implementations are in "inferences" folder and the
  inference implementations return a posterior approximation that the GP models use.

- "acquisitions" folder contain the implementations for the acquisition functions.

- sigopt_functions.py contains all the functions used as examples in the paper.
  The functions are mostly copied from
  https://github.com/sigopt/evalset/blob/master/evalset/test_funcs.py

- utils.py contains useful methods used by many parts of the code


Everything is well commented so reading and understanding the source
code should be easy given that the reader is familiar with the paper
and BO in general.