# AI-ML-Applications-on-Gadi_Astronomy

There are 2 parts and notebooks in this session. We will be using `part1_emulator.ipynb` and `part2_inference.ipynb` to demonstrate how one can use neural networks to emulate a complex model and how to use the emulator in Bayesian inference. The folder, `answers`, contains the same 2 notebooks with all blocks filled.

------------
Requirments:
------------

1. Required modules on Gadi:\
`module purge`\
`module load gcc/11.1.0 openmpi/4.0.7 gsl/2.6 hdf5/1.10.5p cmake/3.18.2 python3/3.8.5`

2. Required Python packages:\
`pip install numpy emcee scipy corner matplotlib tensorflow pymultinest mpi4py tqdm`

3. MultiNest (optional), a required package to be built from source:
        - git clone https://github.com/JohannesBuchner/MultiNest
        - cd MultiNest/build
        - cmake -DCMAKE_INSTALL_PREFIX="WhereToInstall" ..; make install
 P.S. to solve the *bug* with `cmake` on Gadi using `ccmake`
        - hit t to toggle advanced mode
        - input in `MPI_Fortran_F77_HEADER_DIR` with `/apps/openmpi-mofed5.5-pbs2021.1/4.0.7/include`
        - input in `MPI_Fortran_MODULE_DIR` with `/apps/openmpi-mofed5.5-pbs2021.1/4.0.7/lib`
        - hit `c` to configure and then `g` to generate makefile
        - if configuration at step 4 fails, delete the input for entry `m` then go to step 4 again
