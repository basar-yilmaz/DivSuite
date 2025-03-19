#!/bin/bash

configs=(
    "configs/config_sy.yaml"
    "configs/config_swap.yaml"
    "configs/config_msd.yaml"
    "configs/config_motley.yaml"
    # "configs/config_mmr.yaml"
    # "configs/config_bswap.yaml"
)

mkdir -p logs1  # Create logs1 directory
> pids.txt       # Clear pids file

for config in "${configs[@]}"; do
    echo "Running experiment with config: $config"
    nohup python main.py --config "$config" --threshold_drop 0.1 > "logs1/$(basename $config .yaml).log" 2>&1
    echo $! >> pids.txt  # Save PID
done

echo "All experiments started independently! PIDs saved in pids.txt"
