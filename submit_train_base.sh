#!/bin/bash

# Define the parameters
params=(
  "cfg2 mc eb"
  "cfg2 data eb"
)

# Loop over the parameters
for param in "${params[@]}"; do
  IFS=' ' read -r -a param_array <<< "$param"
  param1=${param_array[0]}
  param2=${param_array[1]}
  param3=${param_array[2]}

  # Create a batch script for this set of parameters
  batch_script=$(sed "s/{param1}/$param1/g; s/{param2}/$param2/g; s/{param3}/$param3/g" jobs/train_base.sub)

  # Submit the batch script to Slurm
  echo "$batch_script" | sbatch
done
