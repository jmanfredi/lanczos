# lanczos
Lanczos Tridiagonalization on a GPU

Lanczos tridiagonalization is a crucial part of understanding the giant and sparse Hamiltonian matrices used in theoretical nuclear structure calculations. The Lanczos algorithm is a Krylov subspace method, meaning that repeated matrix-vector products are used to construct an appropriate basis. Such products can in principle be sped up using a GPU. 

Although modern GPUs do not have enough on-board memory for Lanczos tridiagonalization to work for state-of-the-art shell model matrices, future generations of GPUs may be up to the task. Here, I explore Lanczos tridiagonalization using several different sparse matrix-vector algorithms for the GPU.

Author: Juan Manfredi

Free use of code permitted, with citation.
