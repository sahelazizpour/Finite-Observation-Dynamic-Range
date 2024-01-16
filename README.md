# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## Installation
conda create -n finite-observation python=3.11
conda activate finite-observation
conda install -c conda-forge jax
pip install flax
pip install matplotlib pandas tqdm h5py

pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install scikit-learn

pip install numba
pip install dask?

# to activate git filters that allow keeping database in repository without repository to blow up
git config --local include.path .gitconfig 