# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## Future Idea:
Train neural network as a non-parametric approximation (avoid going via beta distribution)

## Installation
conda create -n finite-observation python=3.11
conda activate finite-observation
conda install -c conda-forge jax
pip install flax
pip install matplotlib pandas tqdm h5py

pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install scikit-learn

pip install numba
pip install dask, distributed

# if database result is to be put into repository do this by dumping the sqlite file with
sqlite3 file.db .dump > db_file.txt
