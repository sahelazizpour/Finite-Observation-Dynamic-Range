#!bin/bash
# internal note: sequentual queue rostam accessed via sohrab

#$ -S /bin/bash
#$ -N fodr
#$ -q rostam.q
#$ -cwd
#$ -o /scratch02.local/johannes/projects/sahel_finite-observation-dynamic-range/log/
#$ -e /scratch02.local/johannes/projects/sahel_finite-observation-dynamic-range/log/
#$ -t 1-595

seed=1009

conda activate finite-observation
exe="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/scripts/run_simulation.py"
file_db="/data.nst/johannes/projects/sahel_finite-observation-dynamic-range/simulations.db"

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
