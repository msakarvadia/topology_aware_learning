#!/bin/bash 
#PBS -l select=100
#PBS -l walltime=06:00:00
#PBS -q prod
#PBS -l daos=daos_user
#PBS -l filesystems=home:flare:daos_user
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N daos_bd_scheduler
#PBS -r y 

export http_proxy="http://proxy.alcf.anl.gov:3128"

module use /soft/modulefiles
module load daos

DAOS_POOL=AuroraGPT
DAOS_CONT=decML

#If the container already exists this won't matter
daos container create --type POSIX ${DAOS_POOL} ${DAOS_CONT} --properties rd_fac:1

# make temp dir
mkdir /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} -p

# To mount on login node
#start-dfuse.sh -m /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} --pool ${DAOS_POOL} --cont ${DAOS_CONT}

# To mount on all compute nodes
#launch-dfuse.sh /tmp/${USER}/${DAOS_POOL}:${DAOS_CONT}
launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
mount | grep dfuse # To confirm if its mounted

# List the content of the container
ls /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}

# move into the daos file system
cd /tmp/${DAOS_POOL}/${DAOS_CONT}


EXPERIMENT_DIR=/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
TOPO_FILE=${EXPERIMENT_DIR}src/create_topo/topology/topo_1.txt

module load frameworks
source ${EXPERIMENT_DIR}env/bin/activate

pip list

echo ${EXPERIMENT_DIR}

pwd

python ${EXPERIMENT_DIR}/src/experiments/bd_scheduler.py --rounds 40

touch tmp.txt
# To unmount
fusermount3 -u /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
