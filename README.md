
# INSTALLATION

The environment for CusADi 2.0 is managed and built with `conda` (whereas v1.0 was built with `pip`), due to dependencies with `pinocchio` and `casadi`.

CusADi can be installed standalone to design and/or parallelize optimal controllers, or as part of a submodule in a larger project (e.g. [IsaacLab](https://github.com/isaac-sim/IsaacLab)) for reinforcement learning.

### Requirements:
- >= Ubuntu 22.04
- >= Python 3.11
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- CUDA Toolkit and compiler
- >= cuDSS 0.7.0

### Installation:

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install viser
pip install yourdfpy
pip install -e .



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
    


