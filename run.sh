#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

#TODO: assumes xgboost, add option for selecting other models (when ready)
helpFunction()
{
   echo ""
   echo "Usage: $0 -n numclients"
   echo -e "\t-n Number of federated clients"
   exit 1 # Exit script after printing help
}

while getopts "n:" opt
do
   case "$opt" in
      n ) numclients="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$numclients" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

echo "Starting server"
python xgboost/server.py --pool-size="$numclients" --num-clients-per-round="$numclients" --num-evaluate-clients="$numclients" &
sleep 10  # Sleep for 10s to give the server enough time to start and dowload the dataset

for i in $(seq 0 $((numclients - 1))); do
    echo "Starting client $i"
    python xgboost/client.py --partition-id="${i}" --num-partitions="$numclients" &
done

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait