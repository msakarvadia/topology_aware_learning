
export JOBNAME=parsl.decentral_train.block-0.1744580488.2916284
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py --debug --max_workers_per_node=12 -a 10.113.40.105,10.113.40.120,127.0.0.1,10.113.40.167,10.113.40.108,10.113.40.140,10.113.40.110,10.113.40.118,10.113.40.163,10.115.24.3 -p 0 -c 1.0 -m None --poll 10 --task_port=44764 --result_port=44487 --cert_dir None --logdir=/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments/runinfo/000/decentral_train --block_id=0 --hb_period=15  --hb_threshold=120 --drain_period=None --cpu-affinity list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203  --mpi-launcher=mpiexec --available-accelerators 0.0 0.1 1.0 1.1 2.0 2.1 3.0 3.1 4.0 4.1 5.0 5.1
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
