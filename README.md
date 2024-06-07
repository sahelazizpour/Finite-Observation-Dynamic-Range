# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## TODO:
* make notebooks more consistent by moving analysis pipelines to analysis.py and only calling them from the notebook. Maybe they can then be actually merged?
* Reply to # SAHEL comments
* Clean up the test notebooks
* clean up production notebooks!
    * Make one notebook to plot the figures that only load files that are available under ./results/
    * 

## Installation
```
conda create -n finite-observation python=3.11
conda activate finite-observation
conda install -c conda-forge jax
pip install flax
pip install matplotlib pandas tqdm h5py scipys
```

```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install scikit-learn
```

```
pip install numba
pip install dask, distributed
```

# if database result is to be put into repository do this by dumping the sqlite file with
sqlite3 file.db .dump > db_file.txt
