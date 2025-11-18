
# INSTALLATION

The environment for CusADi 2.0 is managed and built with `conda` (whereas v1.0 was built with `pip`), due to dependencies on `pinocchio` and `casadi`.

CusADi can be installed standalone to design and/or parallelize optimal controllers, or as part of a submodule in a larger project (e.g. [IsaacLab](https://github.com/isaac-sim/IsaacLab)) for reinforcement learning.

### Requirements:
- Ubuntu 20.04 or higher
- Python 3.11 or higher
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- Optional (for GPU parallelization):
    - NVIDIA GPU
    - CUDA Toolkit and compiler
    - cuDSS 0.7.0

### Standalone:
    cd $CUSADI_ROOT
    conda env create -f environment.yml
    conda activate cusadi

    # Without GPU parallelization
    pip install -e .

    # With GPU parallelization
    pip install -e --install-option="--with-cuda" .
    
### Submodule:
    cd $PARENT_PROJECT_ROOT
    git submodule add https://github.com/se-hwan/cusadi .
    conda install -n $PARENT_CONDA_ENV -y -f cusadi/environment.yml
    
    # Without GPU parallelization
    pip install -e cusadi
    
    # With GPU parallelization
    pip install -e --install-option="--with-cuda" cusadi


