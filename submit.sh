#!/bin/bash
for SEED in $(seq 0 2); do
  python subspace_algorithm/run_dataset.py --path configs/nanoparticle_synthesis/library_noise_0.01_no_requery.json --seed $SEED
  python subspace_algorithm/run_mobo.py --path configs/nanoparticle_synthesis/library_noise_0.01_no_requery.json --seed $SEED
done