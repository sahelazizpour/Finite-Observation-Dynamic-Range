#!bin/bash
# internal note: parallel queue SAM accessed via thamineh

#$ -S /bin/bash
#$ -N fodr
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o /data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/logs/
#$ -e /data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/logs/
#$ -t 1-21

seed=1009

conda activate finite-observation
exe="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/scripts/run_analysis_simulation.py"
file_db="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/simulations.db"

#windows=(1 10 100 1000 10000)
log_windows=($(seq 0 0.2 4))
num_windows=${#log_windows[@]}
echo "total number of jobs: $((num_windows))"

log_window=${log_windows[$SGE_TASK_ID - 1]}
window=$(python -c "print(10**$log_window)")

echo "submit script with seed=$seed, window=$window"

path='./dat'
mkdir -p $path

python $exe --window $window --seed $seed --path $path --database $file_db