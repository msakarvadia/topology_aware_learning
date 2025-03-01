#!/bin/bash 
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l daos=daos_user
#PBS -l filesystems=home:flare:daos_user
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N daos_lm
#PBS -r y 

module use /soft/modulefiles
module load daos

export DAOS_POOL=AuroraGPT
export DAOS_CONT=decML

#daos container create --type POSIX ${DAOS_POOL} ${DAOS_CONT} --properties rd_fac:1

# make temp file
mkdir /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} -p


# To mount
start-dfuse.sh -m /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT} --pool ${DAOS_POOL} --cont ${DAOS_CONT}
mount | grep dfuse # To confirm if its mounted

# List the content of the container
ls /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}

# move into the file system
cd /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}

# Copy a file to the container
#echo "hello" > hello.txt
#cp hello.txt /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/
#cat /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}/hello.txt



# To unmount
fusermount3 -u /tmp/${USER}/${DAOS_POOL}/${DAOS_CONT}
