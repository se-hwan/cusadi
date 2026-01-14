
# INSTALLATION

The environment for CusADi 2.0 is managed and built with `conda` (whereas v1.0 was built with `pip`), due to dependencies with `pinocchio` and `casadi`.

CusADi can be installed standalone to design and/or parallelize optimal controllers, or as part of a submodule in a larger project (e.g. [IsaacLab](https://github.com/isaac-sim/IsaacLab)) for reinforcement learning.

### Requirements:
- NVIDIA CUDA-compatible GPU
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- >= Ubuntu 22.04
- >= Python 3.11
- >= 13.1 CUDA Toolkit and compiler: https://developer.nvidia.com/cuda-downloads
- >= cuDSS 0.7.0: https://developer.nvidia.com/cudss-downloads

### Installation:

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install viser
pip install yourdfpy
pip install -e .

export PATH="/usr/local/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"

### Standalone:
    cd $CUSADI_ROOT
    conda env create -f environment.yml
    conda activate cusadi
    pip install -e .

### Submodule:
    cd $PARENT_PROJECT_ROOT
    git submodule add https://github.com/se-hwan/cusadi .
    conda env update --name $PARENT_CONDA_ENV --file cusadi/environment.yml
    pip install -e cusadi


### Efficiency improvements:
- DAG analysis of function structure, check if finite set (~100?) of work temporary variables can be used at a time ("liveness-based computation", reorder instructions?)
- CUDA recorded graphs with cuBLAS for linear algebra and parallel reduction functions (sum, max, etc.) with carefully chosen casadi MX operations (but how to deal with sparsity?)
- Manually write Ruiz equilibration CUDA kernel for KKT matrix scaling, casadi mmax is inefficient compared to parallel reduction (but through sparsity, maybe have to pass nonzero elements @ each column)
- JAX implementation and bridge (memory issue with third-party jaxadi implementation for large functions)