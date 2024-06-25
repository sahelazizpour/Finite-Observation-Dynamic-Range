# Finite-Observation-Dynamic-Range

Simulation and analysis code in order to assess the impact of finite observation times on the concept of dynamics range.

## Installation (conda)
For reproduction it should be sufficient to install the working environment via
```
conda env create -f environment.yml
conda activate finite-observation
```

Alternatively, a step-by-step installation using only conda was done as follows
```
conda create -n finite-observation python=3.10
conda activate finite-observation
conda install pytorch torchvision torchaudio -c pytorch
conda install ipykernel matplotlib pandas tqdm h5py scipy 
conda install scikit-learn=1.3.2 dask distributed
```

If pure conda does not work, there is an option to combine with pip
```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

#conda env export --no-builds | grep -v "prefix" > environment.yml

# if database result is to be put into repository do this by dumping the sqlite file with
sqlite3 file.db .dump > db_file.txt
