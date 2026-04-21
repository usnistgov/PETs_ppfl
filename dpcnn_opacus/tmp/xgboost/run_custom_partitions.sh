#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# check if "logs" directory is present, create it if not
if [ ! -d "logs" ] ;
    then
    mkdir "logs" ;
fi

n_clients=5

python server.py --num-rounds 2 --pool-size $n_clients \
--num-clients-per-round $n_clients  --num-evaluate-clients $n_clients &
sleep 10  # Sleep for 10s to give the server enough time to start

for i in $(seq 0 $(($n_clients - 1))); do
    echo "Starting client $i"
    python client.py --partition-id "${i}"  --client-id "${i}"\
    --num-partitions $n_clients \
    --centralised-eval \
    --data-partitions-file "../genetic_plant_data/ppfl_FlC_5clients_2024_12_31.npz"&
done

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
