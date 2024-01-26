#!bin/bash
# internal note: sequentual queue rostam accessed via sohrab

#$ -S /bin/bash
#$ -N fodr
#$ -q rostam.q
#$ -cwd
#$ -o /scratch02.local/johannes/projects/sahel_finite-observation-dynamic-range/log/
#$ -e /scratch02.local/johannes/projects/sahel_finite-observation-dynamic-range/log/
#$ -t 1-595


conda activate finite-observation
exe="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/1_run_simulation.py"
file_db="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/0_update/simulations.db"

seed=1000

#bash equivalent to numpys hs=np.logspace(-4, 1, 101) going from 1e-4 to 10

loghs=($(seq -6.5 0.25 2))
#loghs=($(seq 1.25 0.25 2))#$ -t 1-68
numhs=${#loghs[@]}
loges=($(seq 0 -0.25 -4))
numes=${#loges[@]}
echo "total number of jobs: $((numhs*numes))"

idh=$(( (SGE_TASK_ID-1)%numhs ))
ide=$(( (SGE_TASK_ID-1)/numhs ))

logh=${loghs[$idh]}
loge=${loges[$ide]}

echo "submit script with seed=$seed, log(h)=$logh, log(1-lamda)=$loge"

path='/scratch02.local/johannes/projects/sahel_finite-observation-dynamic-range/'
mkdir -p $path

python $exe --log10_eps ${loge} --log10_h ${logh} --seed $seed --path $path --database $file_db