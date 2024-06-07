# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## TODO:
* Reply to # SAHEL comments
* match run_analysis_simulations to be the same as in analytic solutions
* clean up production notebooks!
    * plot_workflow (streamline)
    * train_neural_network -> train_neural_network_to_approximate_beta
    * fit_beta_to_data -> fit_beta_to_simulations
    * 
* Clean up the test notebooks

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
pip install dask distributed
```

# if database result is to be put into repository do this by dumping the sqlite file with
sqlite3 file.db .dump > db_file.txt
