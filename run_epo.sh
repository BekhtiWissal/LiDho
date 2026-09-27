#!/bin/bash
# Run EPO (generalization) scenarios on SMACv2 for a given algorithm
# Usage: bash run_epo.sh <algo> [num_seeds] [map]
# Example: bash run_epo.sh godeal 1 sc2_gen_protoss_epo
# Maps: sc2_gen_protoss_epo, sc2_gen_terran_epo, sc2_gen_zerg_epo
# Algorithms: godeal, qmix, ippo, mappo, vdn, ...

ALGO=${1:-godeal}
NUM_SEEDS=${2:-1}
MAPS=(${3:-sc2_gen_protoss_epo})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SC2PATH="${SC2PATH:-"${SCRIPT_DIR}/3rdparty/StarCraftII"}"

echo "Algorithm : ${ALGO}"
echo "Seeds     : 0...$((NUM_SEEDS-1))"
echo "Maps      : ${MAPS[*]}"
echo "SC2PATH   : ${SC2PATH}"
echo ""

for map in "${MAPS[@]}"; do
    for i in $(seq 0 $((NUM_SEEDS - 1))); do
        python "${SCRIPT_DIR}/src/main.py" \
            --config="${ALGO}" \
            --env-config=sc2v2_epo \
            with env_args.map_name="${map}" \
                 seed="${i}" &
        echo "Launched: algo=${ALGO} map=${map} seed=${i} (PID $!)"
        sleep 2
    done
done

wait
echo "All EPO runs finished."
