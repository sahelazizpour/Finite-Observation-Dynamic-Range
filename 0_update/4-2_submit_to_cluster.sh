#!bin/bash
# internal note: parallel queue SAM accessed via thamineh

#$ -S /bin/bash
#$ -N fodr
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o /data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/logs/
#$ -e /data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/logs/
#$ -t 1-5

conda activate finite-observation
exe="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/4_run_analysis_simulation.py"
file_db="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/simulations.db"

windows=(1 10 100 1000 10000)
window=${windows[$SGE_TASK_ID-1]}

seed=1005

echo "submit script with seed=$seed, window=$window"

path='/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/dat/'
mkdir -p $path

python $exe --window $window --seed $seed --path $path --database $file_db