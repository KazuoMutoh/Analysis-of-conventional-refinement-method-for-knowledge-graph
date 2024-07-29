#!/bin/bash

# options for batch job execution.
# for all options, please check https://docs.abci.ai/ja/job-execution/#job-execution-options 

#$ -l rt_G.small=1
#$ -l h_rt=15:00:00
#$ -m a
#$ -m b
#$ -m e
#$ -j y
#$ -o ./logs_job/script_pykeen_fb15k237_transe.log
#$ -cwd

source /etc/profile.d/modules.sh
source /home/acg16558pn/kg_20240423/bin/activate
module load cuda/12.1
module load python/3.10
python script_pykeen_hpo.py -m transe -d fb15k237 -n 30 -o ./reports/20240623/pykeen_hpo_fb15k237_transe
