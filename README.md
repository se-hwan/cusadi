
# INSTALLATION

The environment for CusADi 2.0 is managed and built with `conda` (whereas v1.0 was built with `pip`), due to dependencies with `pinocchio` and `casadi`.

CusADi can be installed standalone to design and/or parallelize optimal controllers, or as part of a submodule in a larger project (e.g. [IsaacLab](https://github.com/isaac-sim/IsaacLab)) for reinforcement learning.

### Requirements:
- NVIDIA CUDA-compatible GPU
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Ubuntu 22.04
- $\geq$ Python 3.11
- $\geq$ 13.1 CUDA Toolkit and compiler: https://developer.nvidia.com/cuda-downloads
- $\geq$ cuDSS 0.7.0: https://developer.nvidia.com/cudss-downloads

### Installation:
```
```

export PATH="/usr/local/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"

### Standalone:
```
cd $CUSADI_ROOT
conda env create -f environment.yml
conda activate cusadi
pip install -r requirements.txt
pip install -e .
```

### Submodule:
```
cd $PARENT_PROJECT_ROOT
git submodule add https://github.com/se-hwan/cusadi .
conda env update --name $PARENT_CONDA_ENV --file cusadi/environment.yml

# Keep IsaacLab / Isaac Sim as the source of truth for shared packages.
# Do not let a broad pip install rewrite its pinned dependencies.
pip install --no-deps -e cusadi
pip install ninja


# Optional visualization extras. Install only if needed, and prefer a separate
# environment if they trigger resolver conflicts with IsaacLab.
# pip install viser yourdfpy seaborn
```

! May be necessary
In `env_vars.sh` located at `~/miniconda3/envs/$ENV_NAME/etc/conda/activate.d`
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


### TO DO:
- [ ] Unit tests
    - [ ] Parallelization (single and multiple functions)
    - [ ] cuDSS interface
    - [ ] ADMM implementation (compare against OSQP)
    - [ ] Pinocchio model
    - [ ] Many more...
- [ ] Rename solvers to more general "ADMM" and "Barrier", not strictly OSQP
- [ ] Casadi function storage and loading from OptimizationProblem object
- [ ] Options for various barrier functions (max, log, etc.)
- [ ] Add barrier options to config (mu, delta, coefficients for different barriers, etc.)
- [ ] Refactor qp_backends.py, too large, separate
- [ ] Refactor sqp_interface.py, add simple line search options
- [ ] Refactor ipopt_interface.py
- [ ] Cleanup pinocchio_model.py, reevaluate necessity of model.py
- [ ] Cleanuip cusadi/utils folder, necessary? Clean up symbolic.py
- [ ] Parse through to_sort folder


### Efficiency improvements:
- DAG analysis of function structure, group parallelizable operations (overhead from synchronization?)
- Chunk work variable temporaries (chunk of variables in registers (local kernel vars), chunk of variables in shared_memory. minimize r/w operations to global memory `work` tensor)
- CUDA recorded graphs with cuBLAS for linear algebra and parallel reduction functions (sum, max, etc.) with carefully chosen casadi MX operations (but how to deal with sparsity?)
- Manually write Ruiz equilibration CUDA kernel for KKT matrix scaling, casadi mmax is inefficient compared to parallel reduction (but through sparsity, maybe have to pass nonzero elements @ each column)
- JAX implementation and bridge (memory issue with third-party jaxadi implementation for large functions)