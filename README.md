# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## TODO:
* src code
    * simulation: Reply to # SAHEL comments, check if jit is helpful or if numba can be removed
    * explain the origin of the timescales
* clean up production notebooks!
    * plot_workflow (streamline)
    * train_neural_network -> train_neural_network_to_approximate_beta
    * fit_beta_to_data -> fit_beta_to_simulations
* Clean up the test notebooks
* RERUN Simulation analysis pipeline
    1) analysis of simulation using old neural network (new text-file order!)
    2) refit simulations (new neural network without sklearn?)
    3) rerun simulaitons?
* Fig. 1:
    * keep a in general (but redo in affintiy with nice colors)
    * redo panel b and c with the actual output
* Fig. 2:
    * Update to match with Fig. workflow

## Installation (Mac M1)
```
micromamba create -n finite-observation python=3.11
micromamba activate finite-observation
micromamba install ipykernel matplotlib pandas tqdm h5py scipy 
pip3 install --pre torch torchvision torchaudio #--extra-index-url https://download.pytorch.org/whl/nightly/cpu
micromamba install scikit-learn=1.3.2 dask distributed
```

## Installation (Linux)
```
conda create -n finite-observation python
conda activate finite-observation
conda install ipykernel matplotlib pandas tqdm h5py scipy 
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
conda install scikit-learn=1.3.2 dask distributed
```

# if database result is to be put into repository do this by dumping the sqlite file with
sqlite3 file.db .dump > db_file.txt
